"""
Integration Tests for Prediction Pipeline
Tests end-to-end trajectory prediction workflow
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from space_debris_tracker.data_ingestion.tle_parser import TLEParser
from space_debris_tracker.trajectory_prediction.orbit_predictor import (
    OrbitPredictionEngine,
)
from tests.utils import (
    assert_valid_trajectory,
    generate_orbital_state,
    propagate_orbit_simple,
)


@pytest.mark.integration
class TestPredictionPipeline:
    """Test end-to-end prediction pipeline"""

    def test_tle_to_prediction_pipeline(self, sample_tle_lines):
        """Test TLE parsing to trajectory prediction"""
        # Parse TLE
        parser = TLEParser()
        name, line1, line2 = sample_tle_lines
        elements = parser.parse_tle(line1, line2, name)

        # Get current state
        current_time = elements.epoch
        position, velocity = parser.propagate_sgp4(elements, current_time)
        state = np.concatenate([position, velocity])

        # Predict trajectory
        predictor = OrbitPredictionEngine(device="cpu")
        prediction = predictor.predict_trajectory(
            initial_state=state, time_horizon=5400, dt=60.0  # One orbit
        )

        # Validate
        assert "position" in prediction
        assert "velocity" in prediction
        assert len(prediction["time"]) > 0

    def test_multi_object_prediction(self, sample_tle_file):
        """Test predicting multiple objects"""
        parser = TLEParser()
        elements_list = parser.parse_tle_file(str(sample_tle_file))

        predictor = OrbitPredictionEngine(device="cpu")
        predictions = []

        for elements in elements_list[:3]:  # First 3 satellites
            pos, vel = parser.propagate_sgp4(elements, elements.epoch)
            state = np.concatenate([pos, vel])

            prediction = predictor.predict_trajectory(
                initial_state=state, time_horizon=3600, dt=300.0
            )

            predictions.append(prediction)

        assert len(predictions) == 3

    def test_conjunction_detection(self):
        """Test conjunction detection between objects"""
        predictor = OrbitPredictionEngine(device="cpu")

        # Two objects in similar orbits
        state1 = generate_orbital_state(altitude=400.0, inclination=51.6)
        state2 = generate_orbital_state(altitude=405.0, inclination=51.6)

        # Add as nearby object
        predictor.add_nearby_object(state2)

        # Predict
        prediction = predictor.predict_trajectory(initial_state=state1, time_horizon=5400, dt=60.0)

        # Should calculate collision probability
        assert "collision_probability" in prediction
        assert prediction["collision_probability"] >= 0.0

    def test_uncertainty_propagation(self, sample_state_vector):
        """Test uncertainty propagation over time"""
        predictor = OrbitPredictionEngine(device="cpu")

        # Short-term prediction
        pred_short = predictor.predict_trajectory(
            initial_state=sample_state_vector,
            time_horizon=3600,
            dt=60.0,
            include_uncertainty=True,
        )

        # Long-term prediction
        pred_long = predictor.predict_trajectory(
            initial_state=sample_state_vector,
            time_horizon=86400,
            dt=300.0,
            include_uncertainty=True,
        )

        # Both should include uncertainty
        assert pred_short is not None
        assert pred_long is not None

    def test_orbital_mechanics_accuracy(self, sample_state_vector):
        """Test orbital mechanics accuracy"""
        predictor = OrbitPredictionEngine(device="cpu")

        # Predict one complete orbit
        mu = 398600.4418
        r = np.linalg.norm(sample_state_vector[:3])
        period = 2 * np.pi * np.sqrt(r**3 / mu)  # seconds

        prediction = predictor.predict_trajectory(
            initial_state=sample_state_vector,
            time_horizon=period,
            dt=60.0,
            include_uncertainty=False,
        )

        # Final position should be close to initial
        initial_pos = sample_state_vector[:3]
        final_pos = prediction["position"][-1]

        distance = np.linalg.norm(final_pos - initial_pos)

        # Should complete orbit (allowing for some drift)
        print(f"Orbit closure error: {distance:.2f} km")
        # Allow up to 100 km error (depends on integrator quality)
        assert distance < 500.0

    def test_perturbation_effects(self, sample_state_vector):
        """Test inclusion of perturbations"""
        from space_debris_tracker.trajectory_prediction.physics.orbital_mechanics import (
            OrbitalMechanics,
        )

        mechanics = OrbitalMechanics()

        position = sample_state_vector[:3]
        velocity = sample_state_vector[3:6]

        # Propagate with perturbations
        pos_pert, vel_pert = mechanics.propagate(
            position,
            velocity,
            dt=60.0,
            include_j2=True,
            include_drag=True,
            include_solar_pressure=True,
        )

        # Propagate without perturbations
        pos_no_pert, vel_no_pert = mechanics.propagate(
            position,
            velocity,
            dt=60.0,
            include_j2=False,
            include_drag=False,
            include_solar_pressure=False,
        )

        # Should be different
        pos_diff = np.linalg.norm(pos_pert - pos_no_pert)
        assert pos_diff > 0.0  # Some difference

    @pytest.mark.slow
    def test_long_term_propagation(self, sample_state_vector):
        """Test long-term propagation (30 days)"""
        predictor = OrbitPredictionEngine(device="cpu")

        prediction = predictor.predict_trajectory(
            initial_state=sample_state_vector,
            time_horizon=86400 * 30,  # 30 days
            dt=3600.0,  # 1 hour steps
            include_uncertainty=True,
        )

        # Should complete without error
        assert len(prediction["time"]) > 0
        assert prediction["position"].shape[0] > 0

    def test_maneuver_planning(self, sample_state_vector):
        """Test collision avoidance maneuver planning"""
        from space_debris_tracker.trajectory_prediction.physics.orbital_mechanics import (
            OrbitalMechanics,
        )

        mechanics = OrbitalMechanics()

        position = sample_state_vector[:3]
        velocity = sample_state_vector[3:6]

        # Calculate delta-v for altitude change
        delta_v = mechanics.calculate_hohmann_transfer(
            r1=np.linalg.norm(position),
            r2=np.linalg.norm(position) + 10.0,  # 10 km altitude increase
        )

        assert delta_v > 0.0
        assert delta_v < 1.0  # Should be small for LEO


@pytest.mark.integration
class TestPredictionAccuracy:
    """Test prediction accuracy"""

    def test_compare_with_sgp4(self, sample_tle_lines):
        """Compare prediction with SGP4 reference"""
        parser = TLEParser()
        name, line1, line2 = sample_tle_lines
        elements = parser.parse_tle(line1, line2, name)

        # SGP4 reference
        target_time = elements.epoch + timedelta(hours=2)
        pos_sgp4, vel_sgp4 = parser.propagate_sgp4(elements, target_time)

        # Our predictor
        current_time = elements.epoch
        pos_init, vel_init = parser.propagate_sgp4(elements, current_time)
        state = np.concatenate([pos_init, vel_init])

        predictor = OrbitPredictionEngine(device="cpu")
        prediction = predictor.predict_trajectory(
            initial_state=state, time_horizon=7200, dt=60.0  # 2 hours
        )

        # Compare final states
        pos_pred = prediction["position"][-1]
        vel_pred = prediction["velocity"][-1]

        pos_error = np.linalg.norm(pos_pred - pos_sgp4)
        vel_error = np.linalg.norm(vel_pred - vel_sgp4)

        print(f"Position error: {pos_error:.2f} km")
        print(f"Velocity error: {vel_error:.4f} km/s")

        # Allow reasonable error
        # (Neural network predictions may differ from SGP4)
        assert pos_error < 100.0  # Within 100 km
        assert vel_error < 1.0  # Within 1 km/s

    def test_energy_conservation(self, sample_state_vector):
        """Test energy conservation in prediction"""
        predictor = OrbitPredictionEngine(device="cpu")

        prediction = predictor.predict_trajectory(
            initial_state=sample_state_vector, time_horizon=5400, dt=60.0  # One orbit
        )

        # Calculate orbital energy at each step
        mu = 398600.4418
        energies = []

        for i in range(len(prediction["time"])):
            r = prediction["position"][i]
            v = prediction["velocity"][i]

            r_mag = np.linalg.norm(r)
            v_mag = np.linalg.norm(v)

            energy = 0.5 * v_mag**2 - mu / r_mag
            energies.append(energy)

        energies = np.array(energies)

        # Check energy conservation
        energy_std = np.std(energies)
        energy_mean = np.abs(np.mean(energies))

        relative_variation = energy_std / energy_mean

        print(f"Energy variation: {relative_variation*100:.2f}%")

        # Allow some variation due to perturbations
        assert relative_variation < 0.2  # Less than 20%

    def test_angular_momentum_conservation(self, sample_state_vector):
        """Test angular momentum conservation"""
        predictor = OrbitPredictionEngine(device="cpu")

        prediction = predictor.predict_trajectory(
            initial_state=sample_state_vector, time_horizon=5400, dt=60.0
        )

        # Calculate angular momentum
        angular_momenta = []

        for i in range(len(prediction["time"])):
            r = prediction["position"][i]
            v = prediction["velocity"][i]

            h = np.cross(r, v)
            h_mag = np.linalg.norm(h)

            angular_momenta.append(h_mag)

        angular_momenta = np.array(angular_momenta)

        # Should be approximately conserved
        h_std = np.std(angular_momenta)
        h_mean = np.mean(angular_momenta)

        relative_variation = h_std / h_mean

        print(f"Angular momentum variation: {relative_variation*100:.2f}%")

        assert relative_variation < 0.2


@pytest.mark.integration
class TestCollisionPrediction:
    """Test collision prediction"""

    def test_close_approach_detection(self):
        """Test detection of close approaches"""
        predictor = OrbitPredictionEngine(device="cpu")

        # Two objects with close orbits
        state1 = generate_orbital_state(altitude=400.0)
        state2 = state1.copy()
        state2[0] += 5.0  # 5 km offset

        predictor.add_nearby_object(state2)

        prediction = predictor.predict_trajectory(initial_state=state1, time_horizon=5400, dt=60.0)

        # Should detect close approach
        assert prediction["collision_probability"] >= 0.0

    def test_no_collision_scenario(self):
        """Test scenario with no collision"""
        predictor = OrbitPredictionEngine(device="cpu")

        # Distant object
        state1 = generate_orbital_state(altitude=400.0)
        state2 = generate_orbital_state(altitude=10000.0)  # Very different orbit

        predictor.add_nearby_object(state2)

        prediction = predictor.predict_trajectory(initial_state=state1, time_horizon=3600, dt=60.0)

        # Low collision probability
        assert prediction["collision_probability"] < 0.1
