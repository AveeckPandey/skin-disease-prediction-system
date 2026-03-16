import sagemaker
import boto3
import tarfile
import os
import shutil
from sagemaker.tensorflow import TensorFlowModel

# --- Configuration ---
BUCKET = "mobliebucket"  # Change this if needed
ROLE = "AmazonSageMaker-ExecutionRole-20260316T1352" # The user should provide or we can try to guess/get it
REGION = "us-east-1"
MODEL_NAME = "skin-disease-mobilenet"
ENDPOINT_NAME = "skin-disease-endpoint"

# Paths
LOCAL_MODEL_PATH = os.path.join("..", "model", "skin_disease_mobilenet_model.h5")
INFERENCE_SCRIPT = "inference.py"
PACKAGE_PATH = "model.tar.gz"

def create_tarball(output_filename, model_path, script_path):
    """Create a tar.gz file containing the model and inference script."""
    print(f"Creating {output_filename}...")
    with tarfile.open(output_filename, "w:gz") as tar:
        # SageMaker expects the model file to be in the root
        tar.add(model_path, arcname=os.path.basename(model_path))
        
        # SageMaker expects custom scripts to be in a 'code' directory
        # We'll create a temporary code directory for this
        if not os.path.exists("code"):
            os.makedirs("code")
        shutil.copy(script_path, "code/")
        
        # Add requirements.txt if needed (uncomment if you have one)
        # with open("code/requirements.txt", "w") as f:
        #     f.write("opencv-python-headless\n")
        
        tar.add("code", arcname="code")
    
    # Cleanup temp directory
    shutil.rmtree("code")
    print("Done.")

def deploy():
    # Initialize SageMaker session
    sess = sagemaker.Session()
    
    # Upload to S3
    s3_path = sess.upload_data(path=PACKAGE_PATH, bucket=BUCKET, key_prefix="model")
    print(f"Model uploaded to: {s3_path}")

    # Define the TensorFlow Model
    # Note: framework_version should match what was used to train (e.g., 2.15)
    model = TensorFlowModel(
        model_data=s3_path,
        role=ROLE,
        framework_version="2.15", # Update based on tf.__version__
        entry_point=INFERENCE_SCRIPT
    )

    # Deploy
    print("Deploying endpoint (this may take 5-10 minutes)...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=ENDPOINT_NAME
    )
    print(f"Endpoint deployed: {predictor.endpoint_name}")

if __name__ == "__main__":
    # 1. Create Tarball
    create_tarball(PACKAGE_PATH, LOCAL_MODEL_PATH, INFERENCE_SCRIPT)
    
    # 2. Deploy (Commented out until user confirms Role/Bucket)
    # deploy()
    print("\nNext step: Update the 'ROLE' variable in this script with your SageMaker Execution Role ARN, then run deploy().")
