#---------------------------------------------------
# Targets to run the model pipeline
#---------------------------------------------------
# Download the data
download:
	python -m src.data.download --dataset "coco" --dataset_size "100%"

# Train the model
train:
	python -m src.model.train --dataset "coco" --dataset_size "5%" --batch_size 2 --max_epochs 10 --log_every_n_steps 2 --log_to "wandb"

# Run inference on the test data
inference:
	python -m src.model.inference --image "media/images/1.jpeg" --output_type "file"

# Run all: RUNS ALL SCRIPTS - DEFAULT
all: download train inference
