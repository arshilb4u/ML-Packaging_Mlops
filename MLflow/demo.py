import os
import mlflow
import argparse
import time

def eval(p1,p2):
    output_metric = p1**2 + p2**2
    return output_metric

def main(inp1, inp2):
    with mlflow.start_run(run_name='Demo'):
        mlflow.log_param('param1',inp1)
        mlflow.log_param('param2',inp2)
        metric = eval(inp1, inp2)
        mlflow.log_metric("eval Metric",metric)
        os.makedirs("dummy", exist_ok=True)
        with open('dummy/example.txt', 'wt') as f:
            f.write(f"Artifact created at {time.asctime()}")
        mlflow.log_artifact("dummy")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--param1', "-p1", type=int, default=5)
    args.add_argument('--param2', "-p2", type=int, default=10)
    parsed_args = args.parse_args()

    main(parsed_args.param1, parsed_args.param2)
