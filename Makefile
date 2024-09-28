#---------------------------------------------------
# Targets to run the model pipeline
#---------------------------------------------------
# Download the data
download:
	python -m src.data.download

# Train the model
train:
	python -m src.model.trainer

# Run inference on the test data
test:
	python -m src.model.inference

# Run all: RUNS ALL SCRIPTS - DEFAULT
all: download train test
