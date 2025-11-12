# Imagen base ligera con Python
FROM python:3.10-slim

# Configura el directorio de trabajo
WORKDIR /app

# Evita que Python genere archivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copia los archivos del proyecto
COPY . /app

# Instala las dependencias de Python
RUN pip install -r requirements.txt

# Exponer el puerto del servidor Dash
EXPOSE 8050

# Comando para ejecutar la app
CMD ["python", "dashboard.py"]