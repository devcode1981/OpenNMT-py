FROM pytorch/pytorch:latest
LABEL maintainer "Stig-Arne Gronroos <stig-arne.gronroos@aalto.fi>"

# Install OpenNMT-py
RUN git clone https://github.com/OpenNMT/OpenNMT-py.git \
    && cd OpenNMT-py \
    && pip install -r requirements.txt \
    && pip install flask \
    && pip install pyonmttok \
    && python setup.py install

# Add unprivileged user
RUN groupadd -r onmt && useradd -r -g onmt onmt

USER onmt

# Autorun the server
ENTRYPOINT [ "/workspace/OpenNMT-py/server.py" ]
CMD [ "--config=available_models/mmod.conf.json" ]
