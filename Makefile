.PHONY: test clean

# Include reusable venv setup from parent directory
include ../venv.mk

test: $(VENV_BIN)activate
	$(PYTEST)

clean:
	rm -rf __pycache__ .pytest_cache .eggs *.egg-info dist build .uv-cache .venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
