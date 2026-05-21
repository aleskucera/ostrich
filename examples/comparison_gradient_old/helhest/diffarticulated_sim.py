"""Helhest optimization using diffarticulated (ICML 2021).

Usage:
    python examples/comparison_gradient/helhest/diffarticulated_sim.py --save results/diffarticulated.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

def run_in_docker(args):
    """Orchestrate running the benchmark inside the 'diffarticulated-run' container."""
    container_name = "diffarticulated-run"
    
    cmd = [
        "docker", "exec", "-i", container_name, 
        "python3", "python/helhest_benchmark.py"
    ]
    
    print(f"Executing diffarticulated helhest inside Docker container '{container_name}'...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Find the JSON block
        output = result.stdout
        if "---BENCHMARK_RESULTS_START---" in output:
            json_str = output.split("---BENCHMARK_RESULTS_START---")[1].split("---BENCHMARK_RESULTS_END---")[0]
            results = json.loads(json_str.strip())
            
            if args.save:
                os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
                with open(args.save, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Saved results to {args.save}")
            else:
                print(json.dumps(results, indent=2))
        else:
            print("Could not find results marker in Docker output.")
            print(output)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running in Docker: {e.stderr}")
        print(e.stdout)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", metavar="PATH")
    args = parser.parse_args()

    run_in_docker(args)

if __name__ == "__main__":
    main()
