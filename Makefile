#---------------------------------------------------
# Targets to run the model pipeline
#---------------------------------------------------
# Download the data
download:
	python -m src.data.download --dataset "flickr" --dataset_size "100%"

# Train the model
train:
	python -m src.model.train --dataset "flickr" --dataset_size "100%" --batch_size 4 --max_epochs 10

# Run inference on the test data
inference:
	python -m python -m src.model.inference --image "media/images/man_on_bicycle.jpeg" --output_type "file"

# Run all: RUNS ALL SCRIPTS - DEFAULT
all: download train inference
