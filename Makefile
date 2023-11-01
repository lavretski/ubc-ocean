.PNONY: all

clean:
	rm -r dist build ubc_ocean.egg-info dependencies outputs ../submission.csv __pycache__

downl_dep:
	mkdir dependencies && \
	pip download -r requirements_kaggle.txt -d "dependencies"