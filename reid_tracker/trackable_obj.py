class TrackableObj:
    def __init__(self, id, embedding, instance_count_for_matching):
        self.id = id
        self.embedding = [embedding]
        self.instance_count_for_matching = instance_count_for_matching

    def update(self, embedding):
        self.embedding.append(embedding)
        self.embedding = self.embedding[-self.instance_count_for_matching:]
