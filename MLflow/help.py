import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

exp_id = mlflow.create_experiment('Loan_ prediction')

with mlflow.start_run(run_name= 'DecisionTreeClass') as run:
    pass

mlflow.end_run() 
 