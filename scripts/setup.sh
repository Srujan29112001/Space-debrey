#!/bin/bash
# Space Debris Tracking System - Setup Script
# This script initializes the development environment

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "═══════════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if running from project root
if [ ! -f "setup.py" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

print_header "Space Debris Tracking System - Setup"

# ============================================================================
# STEP 1: Check Prerequisites
# ============================================================================
print_header "Step 1/8: Checking Prerequisites"

# Check Python version
print_info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
    print_success "Python $PYTHON_VERSION (>= 3.9 required)"
else
    print_error "Python 3.9+ required, found $PYTHON_VERSION"
    exit 1
fi

# Check for GPU (optional)
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    print_success "GPU detected: $GPU_INFO"
    HAS_GPU=true
else
    print_warning "No NVIDIA GPU detected. CPU-only mode will be used."
    HAS_GPU=false
fi

# Check for Git
if command -v git &> /dev/null; then
    print_success "Git is installed"
else
    print_error "Git is required but not installed"
    exit 1
fi

# Check for Docker (optional)
if command -v docker &> /dev/null; then
    print_success "Docker is installed"
    HAS_DOCKER=true
else
    print_warning "Docker not found. Skipping containerization setup."
    HAS_DOCKER=false
fi

# ============================================================================
# STEP 2: Create Virtual Environment
# ============================================================================
print_header "Step 2/8: Creating Virtual Environment"

if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Skipping creation."
else
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# ============================================================================
# STEP 3: Upgrade pip and Install Core Tools
# ============================================================================
print_header "Step 3/8: Installing Core Tools"

print_info "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
print_success "Core tools upgraded"

# ============================================================================
# STEP 4: Install Dependencies
# ============================================================================
print_header "Step 4/8: Installing Dependencies"

print_info "Installing Python dependencies (this may take several minutes)..."

# Install in stages to handle potential failures
print_info "Installing testing frameworks..."
pip install pytest pytest-cov pytest-asyncio pytest-xdist pytest-timeout pytest-benchmark > /dev/null 2>&1 || print_warning "Some testing packages failed to install"

print_info "Installing code quality tools..."
pip install flake8 black mypy isort pylint bandit safety > /dev/null 2>&1 || print_warning "Some quality tools failed to install"

print_info "Installing core dependencies..."
if [ "$HAS_GPU" = true ]; then
    # Install PyTorch with CUDA support
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1 || print_warning "PyTorch GPU installation failed, trying CPU version..."
    pip install torch torchvision torchaudio > /dev/null 2>&1 || print_warning "PyTorch installation failed"
else
    # Install PyTorch CPU-only
    pip install torch torchvision torchaudio > /dev/null 2>&1 || print_warning "PyTorch installation failed"
fi

print_info "Installing remaining dependencies..."
pip install -r requirements.txt 2>&1 | grep -v "Requirement already satisfied" || print_warning "Some dependencies failed to install"

print_info "Installing project in editable mode..."
pip install -e . > /dev/null 2>&1 || print_warning "Project installation failed"

print_success "Dependencies installed"

# ============================================================================
# STEP 5: Create Directory Structure
# ============================================================================
print_header "Step 5/8: Creating Directory Structure"

DIRECTORIES=(
    "data/telescope_images/raw"
    "data/telescope_images/preprocessed"
    "data/telescope_images/annotated"
    "data/radar_data/raw"
    "data/radar_data/processed"
    "data/tle_data/current"
    "data/tle_data/historical"
    "data/tle_data/validated"
    "data/training/debris_images"
    "data/training/orbits"
    "data/training/conjunctions"
    "data/validation/debris_images"
    "data/validation/orbits"
    "data/validation/conjunctions"
    "data/test/debris_images"
    "data/test/orbits"
    "data/test/conjunctions"
    "models/yolo"
    "models/dino"
    "models/pinn"
    "models/transformer"
    "models/mamba"
    "models/ensemble"
    "logs"
    "cache"
    "checkpoints"
)

for dir in "${DIRECTORIES[@]}"; do
    mkdir -p "$dir"
done

print_success "Directory structure created"

# ============================================================================
# STEP 6: Create Environment Configuration
# ============================================================================
print_header "Step 6/8: Creating Environment Configuration"

if [ -f ".env" ]; then
    print_warning ".env file already exists. Skipping creation."
else
    print_info "Creating .env file from template..."
    cp .env.example .env

    # Generate JWT secret
    JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

    # Update .env with generated secrets
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/your_jwt_secret_key_here_min_32_characters/$JWT_SECRET/" .env
    else
        # Linux
        sed -i "s/your_jwt_secret_key_here_min_32_characters/$JWT_SECRET/" .env
    fi

    print_success ".env file created"
    print_warning "Please edit .env file to configure database passwords and API keys"
fi

# ============================================================================
# STEP 7: Start Docker Services (Optional)
# ============================================================================
print_header "Step 7/8: Starting Docker Services (Optional)"

if [ "$HAS_DOCKER" = true ]; then
    read -p "Do you want to start Docker services (Neo4j, Kafka, Redis, PostgreSQL)? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Starting Docker services..."
        docker-compose up -d

        # Wait for services to be ready
        print_info "Waiting for services to be ready (30 seconds)..."
        sleep 30

        # Check service health
        if docker-compose ps | grep -q "Up"; then
            print_success "Docker services started"
            print_info "Neo4j: http://localhost:7474"
            print_info "Prometheus: http://localhost:9090"
            print_info "Grafana: http://localhost:3000 (admin/admin)"
        else
            print_warning "Some Docker services failed to start. Check with: docker-compose ps"
        fi
    else
        print_info "Skipping Docker services. You can start them later with: docker-compose up -d"
    fi
else
    print_info "Docker not available. Skipping service setup."
fi

# ============================================================================
# STEP 8: Run Validation Tests
# ============================================================================
print_header "Step 8/8: Running Validation Tests"

read -p "Do you want to run validation tests? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Running basic import tests..."
    python3 -c "import space_debris_tracker; print('✓ Package imports successfully')" || print_warning "Package import failed"

    print_info "Running quick unit tests..."
    pytest tests/unit/ -v -m "unit and not gpu and not network" -x --tb=short 2>&1 | head -n 50 || print_warning "Some tests failed"

    print_success "Validation complete"
else
    print_info "Skipping validation tests. You can run them later with: pytest tests/"
fi

# ============================================================================
# Summary
# ============================================================================
print_header "Setup Complete!"

echo ""
echo -e "${GREEN}✓ Environment successfully configured!${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  1. Activate virtual environment:"
echo -e "     ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo "  2. Edit .env file to configure your environment:"
echo -e "     ${YELLOW}nano .env${NC}"
echo ""
echo "  3. Start the API server:"
echo -e "     ${YELLOW}python -m space_debris_tracker.api.server${NC}"
echo ""
echo "  4. Start the dashboard:"
echo -e "     ${YELLOW}streamlit run space_debris_tracker/dashboard/app.py${NC}"
echo ""
echo "  5. Run tests:"
echo -e "     ${YELLOW}pytest tests/unit/ -v${NC}"
echo ""
echo "  6. Generate sample data:"
echo -e "     ${YELLOW}python scripts/generate_sample_data.py --all${NC}"
echo ""
echo "  7. Train models:"
echo -e "     ${YELLOW}python training/train_detector.py --config config.yaml${NC}"
echo ""
echo "Documentation: ${BLUE}https://github.com/Srujan29112001/Space-debrey${NC}"
echo ""

# Deactivate virtual environment for the script
deactivate 2>/dev/null || true

print_success "Setup script completed successfully!"
