#!/usr/bin/env python3
"""CPU synthetic gate then complete frozen v2 inference, without scoring."""
from validate_identity_scale_observations_v2 import main as validate
from run_identity_scale_v2 import main as run

if __name__ == '__main__':
    validate()
    run()
