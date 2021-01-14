class Buffer:
    def __init__(self):
        self.buffer_frames = []
        self.buffer_next_frames = []
        self.buffer_targets_q = []
        self.buffer_actions = []

    def add_experience(self, frame, next_frame, target_q, action):
        self.buffer_frames.append(frame)
        self.buffer_next_frames.append(next_frame)
        self.buffer_targets_q.append(target_q)
        self.buffer_actions.append(action)

    def clear(self):
        self.buffer_frames = []
        self.buffer_next_frames = []
        self.buffer_targets_q = []
        self.buffer_actions = []

    def size(self):
        return len(self.buffer_frames)
