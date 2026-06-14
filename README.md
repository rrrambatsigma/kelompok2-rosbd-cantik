# Proyek Kelompok 2 ROSBD 
PS E:\kelompok2-rosbd-cantik> docker compose up --build -d ingester
PS E:\kelompok2-rosbd-cantik> docker compose logs -f ingester

di pws lain
docker-compose exec kafka kafka-console-consumer --bootstrap-server kafka:9092 --topic flights --from-beginning