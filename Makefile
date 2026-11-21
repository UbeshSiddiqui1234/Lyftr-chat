.PHONY: up down

up:
	@test -f .env || cp .env.example .env
	docker compose up --build -d
	@echo "Frontend: http://localhost:3000"
	@echo "Backend:  http://localhost:8000"

down:
	docker compose down
