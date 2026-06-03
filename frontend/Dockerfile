# Hapus isi folder frontend (jika ada file lama)
Remove-Item -Path frontend\* -Recurse -Force -ErrorAction SilentlyContinue

# Dockerfile sederhana
@'
FROM nginx:alpine
COPY index.html /usr/share/nginx/html/index.html
EXPOSE 80
'@ | Out-File -FilePath frontend\Dockerfile -Encoding ascii

# index.html
@'
<!DOCTYPE html>
<html>
<head><title>ROSBD Pipeline</title></head>
<body>
<h1>Orkestrasi Tools ROSBD Berjalan</h1>
<p>Frontend React akan segera diintegrasikan.</p>
</body>
</html>
'@ | Out-File -FilePath frontend\index.html -Encoding ascii