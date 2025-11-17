# Space Debris Tracking System - Test Suite

Comprehensive testing infrastructure for the Space Debris Tracking & Autonomous Collision Prediction System.

## 📁 Directory Structure

```
tests/
├── conftest.py                 # Pytest configuration and fixtures
├── utils.py                    # Test utilities and helpers
├── unit/                       # Unit tests for individual components
│   ├── test_tle_parser.py
│   ├── test_detector.py
│   ├── test_tracker.py
│   ├── test_orbit_predictor.py
│   ├── test_knowledge_graph.py
│   ├── test_monitoring_agent.py
│   ├── test_data_validator.py
│   └── test_kafka_consumer.py
├── integration/                # Integration tests for workflows
│   ├── test_detection_pipeline.py
│   ├── test_prediction_pipeline.py
│   └── test_api_endpoints.py
├── performance/                # Performance benchmarks
│   ├── test_throughput.py
│   └── test_latency.py
├── fixtures/                   # Test data fixtures
└── data/                       # Test data files
```

## 🚀 Quick Start

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# Performance tests
pytest -m performance

# API tests
pytest -m api

# Exclude slow tests
pytest -m "not slow"

# Exclude GPU tests (when CUDA not available)
pytest -m "not gpu"
```

### Run with Coverage

```bash
pytest --cov=space_debris_tracker --cov-report=html
```

View coverage report in `htmlcov/index.html`

### Run in Parallel

```bash
pytest -n auto
```

## 📊 Test Categories (Markers)

- `unit` - Unit tests for individual components
- `integration` - Integration tests for workflows
- `performance` - Performance and benchmark tests
- `slow` - Tests that take significant time
- `gpu` - Tests requiring GPU/CUDA
- `network` - Tests requiring network access
- `database` - Tests requiring database (Neo4j)
- `kafka` - Tests requiring Kafka
- `api` - API endpoint tests

## 🧪 Test Components

### Unit Tests

#### TLE Parser (`test_tle_parser.py`)
- TLE format validation
- Parsing accuracy
- SGP4 orbit propagation
- Epoch handling
- Derived elements calculation

#### Detector (`test_detector.py`)
- YOLOv7 detection
- DINO v2 zero-shot detection
- Multi-detector fusion
- Object tracking
- 3D characterization

#### Tracker (`test_tracker.py`)
- DeepSORT tracking
- Kalman filtering
- Track management
- Multi-object tracking

#### Orbit Predictor (`test_orbit_predictor.py`)
- PINN prediction
- Transformer refinement
- Mamba2 long-term forecasting
- Physics constraint application
- Uncertainty quantification

#### Knowledge Graph (`test_knowledge_graph.py`)
- Neo4j operations
- Graph queries
- Conjunction tracking
- Historical analysis

### Integration Tests

#### Detection Pipeline (`test_detection_pipeline.py`)
- End-to-end detection workflow
- Multi-frame tracking
- Preprocessing enhancement
- Performance testing

#### Prediction Pipeline (`test_prediction_pipeline.py`)
- TLE to trajectory conversion
- Multi-object prediction
- Conjunction detection
- Orbital mechanics accuracy

#### API Endpoints (`test_api_endpoints.py`)
- REST API endpoints
- GraphQL queries
- WebSocket connections
- Authentication
- Performance

### Performance Tests

#### Throughput (`test_throughput.py`)
- Images per second
- Predictions per second
- Data ingestion rate
- Batch processing

#### Latency (`test_latency.py`)
- Detection latency
- Prediction latency
- API response time
- Database query latency

## 🔧 Configuration

### pytest.ini

Main pytest configuration file with:
- Test discovery patterns
- Coverage settings
- Markers
- Timeout settings

### conftest.py

Shared fixtures:
- Sample TLE data
- Mock images
- State vectors
- Trajectories
- Mock database drivers
- API test clients

### utils.py

Test utilities:
- TLE generators
- Image generators
- State vector generators
- Kafka message generators
- Assertion helpers
- Timing utilities

## 📈 Coverage Goals

- **Overall Coverage**: 80%+
- **Core Components**: 90%+
- **Critical Paths**: 100%

## 🐳 Docker Testing

Run tests in Docker:

```bash
docker-compose run --rm test pytest
```

## 🔄 Continuous Integration

GitHub Actions workflow (`.github/workflows/tests.yml`):
- ✅ Unit tests on Python 3.9, 3.10, 3.11
- ✅ Integration tests with Neo4j and Kafka
- ✅ Code quality checks (Black, Flake8, MyPy)
- ✅ Performance tests
- ✅ Security scanning
- ✅ Coverage reporting to Codecov

## 📝 Writing New Tests

### Test Structure

```python
import pytest
from space_debris_tracker.component import Component

@pytest.mark.unit
class TestComponent:
    """Test Component functionality"""

    def test_basic_functionality(self):
        """Test basic operation"""
        component = Component()
        result = component.process()
        assert result is not None

    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
    ])
    def test_with_parameters(self, input, expected):
        """Test with multiple parameters"""
        assert Component.calculate(input) == expected
```

### Using Fixtures

```python
def test_with_fixture(sample_tle_lines):
    """Test using conftest fixture"""
    name, line1, line2 = sample_tle_lines
    # Use fixture data
```

### Mocking External Dependencies

```python
from unittest.mock import Mock, patch

def test_with_mock():
    """Test with mocked dependency"""
    with patch('module.ExternalService') as mock_service:
        mock_service.return_value.get_data.return_value = {'data': 'test'}
        # Test code
```

## 🔍 Debugging Tests

### Verbose Output

```bash
pytest -v --tb=short
```

### Stop on First Failure

```bash
pytest -x
```

### Run Specific Test

```bash
pytest tests/unit/test_tle_parser.py::TestTLEParser::test_parse_valid_tle
```

### Debug with PDB

```bash
pytest --pdb
```

## 📚 Best Practices

1. **Test Independence**: Each test should be independent
2. **Clear Assertions**: Use descriptive assertion messages
3. **Mock External Services**: Don't rely on external APIs/services
4. **Fixtures for Reusability**: Use fixtures for common test data
5. **Parametrize**: Use `@pytest.mark.parametrize` for multiple cases
6. **Mark Tests**: Use markers to categorize tests
7. **Performance**: Mark slow tests with `@pytest.mark.slow`
8. **Documentation**: Add docstrings to test functions

## 🐛 Common Issues

### Issue: CUDA not available
```bash
pytest -m "not gpu"
```

### Issue: Tests timeout
```bash
pytest --timeout=300  # Increase timeout to 5 minutes
```

### Issue: Database connection failed
```bash
# Ensure Neo4j/Kafka are running
docker-compose up -d neo4j kafka
```

## 📞 Support

For issues or questions about tests:
1. Check test documentation
2. Review conftest.py for available fixtures
3. Check GitHub Actions logs for CI failures
4. Review test markers for categorization

## 📄 License

Tests follow the same license as the main project.
