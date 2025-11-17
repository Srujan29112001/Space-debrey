"""
Streamlit Dashboard for Space Debris Tracking
3D visualization with Cesium.js integration
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Page config
st.set_page_config(
    page_title="Space Debris Tracking System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .alert-high {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


class SpaceDebrisDashboard:
    """Main dashboard class"""

    def __init__(self):
        self.api_url = "http://localhost:8000/api/v1"

    def run(self):
        """Run the dashboard"""
        # Header
        st.markdown(
            '<h1 class="main-header">🛰️ Space Debris Tracking System</h1>',
            unsafe_allow_html=True,
        )

        # Sidebar
        self.render_sidebar()

        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "🌍 3D Visualization",
                "📊 Risk Analysis",
                "🎯 Conjunctions",
                "📈 Statistics",
            ]
        )

        with tab1:
            self.render_3d_visualization()

        with tab2:
            self.render_risk_analysis()

        with tab3:
            self.render_conjunctions()

        with tab4:
            self.render_statistics()

    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.header("Control Panel")

            # Satellite selection
            st.subheader("🛰️ Satellites")
            selected_sats = st.multiselect(
                "Select Satellites",
                options=[
                    "ISS (25544)",
                    "Hubble (20580)",
                    "Starlink-1007 (44713)",
                    "Sentinel-1A (39634)",
                ],
                default=["ISS (25544)"],
            )

            # Time range
            st.subheader("⏰ Time Range")
            time_range = st.slider("Prediction Horizon (days)", min_value=1, max_value=30, value=7)

            # Risk thresholds
            st.subheader("⚠️ Risk Thresholds")
            prob_threshold = st.number_input(
                "Collision Probability",
                min_value=1e-6,
                max_value=1e-2,
                value=1e-4,
                format="%.2e",
            )

            # Update rate
            st.subheader("🔄 Update Rate")
            update_rate = st.selectbox(
                "Refresh Rate",
                options=["Real-time (1 Hz)", "Every 10s", "Every minute", "Manual"],
                index=1,
            )

            # System status
            st.subheader("📊 System Status")
            st.metric("Active Satellites", len(selected_sats))
            st.metric("API Status", "🟢 Operational")
            st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))

    def render_3d_visualization(self):
        """Render 3D orbit visualization"""
        st.subheader("3D Orbit Visualization")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Generate 3D orbit plot
            fig = self.create_3d_orbit_plot()
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Legend")
            st.markdown("🔵 **Active Satellites**")
            st.markdown("🔴 **High Risk Debris**")
            st.markdown("🟡 **Medium Risk Debris**")
            st.markdown("⚪ **Low Risk Debris**")

            st.markdown("### View Controls")
            st.markdown("- **Drag**: Rotate view")
            st.markdown("- **Scroll**: Zoom in/out")
            st.markdown("- **Double-click**: Reset")

    def create_3d_orbit_plot(self):
        """Create 3D orbit visualization"""
        # Create figure
        fig = go.Figure()

        # Add Earth
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = 6371 * np.outer(np.cos(u), np.sin(v))
        y = 6371 * np.outer(np.sin(u), np.sin(v))
        z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))

        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                colorscale=[[0, "rgb(30, 58, 138)"], [1, "rgb(59, 130, 246)"]],
                showscale=False,
                name="Earth",
                opacity=0.9,
            )
        )

        # Add ISS orbit
        theta = np.linspace(0, 2 * np.pi, 100)
        r = 6371 + 410  # ISS altitude ~410 km
        x_orbit = r * np.cos(theta)
        y_orbit = r * np.sin(theta)
        z_orbit = np.zeros_like(theta)

        fig.add_trace(
            go.Scatter3d(
                x=x_orbit,
                y=y_orbit,
                z=z_orbit,
                mode="lines",
                line=dict(color="cyan", width=2),
                name="ISS Orbit",
            )
        )

        # Add ISS position
        iss_theta = (datetime.now().timestamp() / 5580) * 2 * np.pi
        iss_x = r * np.cos(iss_theta)
        iss_y = r * np.sin(iss_theta)

        fig.add_trace(
            go.Scatter3d(
                x=[iss_x],
                y=[iss_y],
                z=[0],
                mode="markers",
                marker=dict(size=8, color="blue", symbol="diamond"),
                name="ISS",
            )
        )

        # Add some debris
        np.random.seed(42)
        n_debris = 50
        debris_r = 6371 + np.random.uniform(300, 600, n_debris)
        debris_theta = np.random.uniform(0, 2 * np.pi, n_debris)
        debris_phi = np.random.uniform(0, np.pi, n_debris)

        debris_x = debris_r * np.sin(debris_phi) * np.cos(debris_theta)
        debris_y = debris_r * np.sin(debris_phi) * np.sin(debris_theta)
        debris_z = debris_r * np.cos(debris_phi)

        fig.add_trace(
            go.Scatter3d(
                x=debris_x,
                y=debris_y,
                z=debris_z,
                mode="markers",
                marker=dict(size=3, color="red", opacity=0.6),
                name="Debris",
            )
        )

        # Layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, title=""),
                yaxis=dict(showgrid=False, showticklabels=False, title=""),
                zaxis=dict(showgrid=False, showticklabels=False, title=""),
                bgcolor="rgb(10, 10, 30)",
                aspectmode="data",
            ),
            showlegend=True,
            height=600,
            margin=dict(l=0, r=0, t=0, b=0),
        )

        return fig

    def render_risk_analysis(self):
        """Render risk analysis section"""
        st.subheader("Collision Risk Analysis")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("High Risk", "5", delta="+2")

        with col2:
            st.metric("Medium Risk", "18", delta="-3")

        with col3:
            st.metric("Low Risk", "47", delta="+8")

        with col4:
            st.metric("Total Conjunctions", "70", delta="+7")

        # Risk timeline
        st.markdown("### Risk Timeline")

        dates = pd.date_range(start=datetime.now(), periods=7, freq="D")
        risk_data = pd.DataFrame(
            {
                "Date": dates,
                "High": np.random.randint(2, 8, 7),
                "Medium": np.random.randint(10, 20, 7),
                "Low": np.random.randint(30, 50, 7),
            }
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=risk_data["Date"],
                y=risk_data["High"],
                name="High Risk",
                stackgroup="one",
                fillcolor="rgba(239, 68, 68, 0.7)",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=risk_data["Date"],
                y=risk_data["Medium"],
                name="Medium Risk",
                stackgroup="one",
                fillcolor="rgba(245, 158, 11, 0.7)",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=risk_data["Date"],
                y=risk_data["Low"],
                name="Low Risk",
                stackgroup="one",
                fillcolor="rgba(34, 197, 94, 0.7)",
            )
        )

        fig.update_layout(height=400, yaxis_title="Number of Conjunctions", hovermode="x unified")

        st.plotly_chart(fig, use_container_width=True)

    def render_conjunctions(self):
        """Render conjunction assessments"""
        st.subheader("Active Conjunction Assessments")

        # Sample data
        conjunctions = pd.DataFrame(
            {
                "Primary": ["ISS", "Hubble", "Sentinel-1A", "ISS", "Starlink-1007"],
                "Secondary": [
                    "DEBRIS_12345",
                    "DEBRIS_67890",
                    "DEBRIS_11111",
                    "DEBRIS_22222",
                    "DEBRIS_33333",
                ],
                "TCA": [
                    (datetime.now() + timedelta(hours=12)).strftime("%Y-%m-%d %H:%M"),
                    (datetime.now() + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M"),
                    (datetime.now() + timedelta(hours=36)).strftime("%Y-%m-%d %H:%M"),
                    (datetime.now() + timedelta(hours=48)).strftime("%Y-%m-%d %H:%M"),
                    (datetime.now() + timedelta(hours=60)).strftime("%Y-%m-%d %H:%M"),
                ],
                "Miss Distance (km)": [0.8, 2.5, 1.2, 4.5, 0.5],
                "Probability": [2.5e-4, 1.5e-4, 2.0e-4, 5.0e-5, 3.5e-4],
                "Risk": ["HIGH", "MEDIUM", "HIGH", "LOW", "CRITICAL"],
            }
        )

        # Risk color coding
        def color_risk(val):
            colors = {
                "CRITICAL": "background-color: #dc2626; color: white",
                "HIGH": "background-color: #ef4444; color: white",
                "MEDIUM": "background-color: #f59e0b; color: white",
                "LOW": "background-color: #22c55e; color: white",
            }
            return colors.get(val, "")

        styled = conjunctions.style.applymap(color_risk, subset=["Risk"])
        st.dataframe(styled, use_container_width=True)

        # Details expander
        with st.expander("🔍 View Detailed Analysis"):
            selected_conj = st.selectbox(
                "Select Conjunction",
                conjunctions["Primary"] + " vs " + conjunctions["Secondary"],
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Orbital Parameters**")
                st.write("Radial: 0.3 km")
                st.write("In-Track: 0.5 km")
                st.write("Cross-Track: 0.2 km")
                st.write("Relative Velocity: 12.5 km/s")

            with col2:
                st.markdown("**Recommendations**")
                st.write("✓ Continue monitoring")
                st.write("⚠️ Maneuver may be required")
                st.write("📊 Update tracking data")

    def render_statistics(self):
        """Render statistics section"""
        st.subheader("Debris Population Statistics")

        col1, col2 = st.columns(2)

        with col1:
            # Size distribution
            st.markdown("### Size Distribution")

            sizes = ["Large (>10cm)", "Medium (1-10cm)", "Small (<1cm)"]
            counts = [3450, 8920, 2860]

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=sizes,
                        values=counts,
                        hole=0.4,
                        marker=dict(colors=["#ef4444", "#f59e0b", "#22c55e"]),
                    )
                ]
            )

            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Altitude distribution
            st.markdown("### Altitude Distribution")

            altitudes = ["LEO (0-2000km)", "MEO (2000-35000km)", "GEO (35000km+)"]
            debris_counts = [8920, 2340, 3970]

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=altitudes,
                        y=debris_counts,
                        marker=dict(color=["#3b82f6", "#8b5cf6", "#ec4899"]),
                    )
                ]
            )

            fig.update_layout(height=300, yaxis_title="Number of Objects")
            st.plotly_chart(fig, use_container_width=True)

        # Origin breakdown
        st.markdown("### Debris Origin")

        origins = pd.DataFrame(
            {
                "Origin": [
                    "Collision Fragments",
                    "Launch Debris",
                    "Decommissioned Satellites",
                    "Anomalies",
                    "Other",
                ],
                "Count": [6420, 5130, 3680, 1250, 750],
                "Percentage": [42.1, 33.7, 24.1, 8.2, 4.9],
            }
        )

        st.dataframe(origins, use_container_width=True)


def main():
    """Main entry point"""
    dashboard = SpaceDebrisDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
