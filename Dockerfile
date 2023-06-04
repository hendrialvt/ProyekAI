FROM python:3.9.13-alpine3.16
COPY . /app
WORKDIR /app
RUN apk add gcc libc-dev
RUN pip3 install --upgrade setuptools
RUN pip3 install --upgrade pip
RUN pip3 install scikit-learn
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install matplotlib
RUN pip3 install tensorflow==2.10.1
RUN pip3 install Flask
CMD python app.py