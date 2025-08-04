build:
	docker build -t mem0-api-server .

run_local:
	docker run -p 8000:8000 -v $(shell pwd):/app --env-file .env mem0-api-server 

.PHONY: build run_local
