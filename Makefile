.PHONY: help init format

help: ## Show help
	@echo "Available commands:"
	@echo "  init		  		- Initialize docker container"
	@echo "  format		  		- Format code"
	@echo "  execute-notebooks	- Execute notebooks"
	@echo "  help      			- Show this help"

init:
	docker build -t derek:latest .
	docker run -it --rm -v $(PWD):/app derek:latest

format:
	pre-commit run -a

execute-notebooks:
	jupyter nbconvert --execute --to notebook --inplace examples/*.ipynb --ExecutePreprocessor.timeout=-1
