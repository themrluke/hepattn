# Plain entry point for run_tracking (validate/test) from outside the experiment directory.
import sys
if __name__ == "__main__":
    sys.path.insert(0, "/shared/projects/hepattn/src/hepattn/experiments/trackml")
    from run_tracking import main
    main()
