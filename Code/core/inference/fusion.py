import time

class FusionEngine:
    def __init__(self):
        # Weights for the scoring formula
        self.W_SGG = 0.5
        self.W_POSE = 0.3
        self.W_IOU = 0.2
        
        # State Machine Memory: {chair_id: {'score': 0.0, 'state': 'EMPTY', 'last_sgg': 0}}
        self.chair_states = {}

    def update_state(self, chair_id, pose_score, iou_score, sgg_score=None):
        """
        Returns the final state ('OCCUPIED' or 'EMPTY') based on weighted fusion.
        """
        if chair_id not in self.chair_states:
            self.chair_states[chair_id] = {'score': 0.0, 'state': 'EMPTY', 'prev_sgg': 0.5}

        # Use previous SGG score if new one isn't available (SGG is slow, runs every 5s)
        if sgg_score is not None:
            self.chair_states[chair_id]['prev_sgg'] = sgg_score
        
        current_sgg = self.chair_states[chair_id]['prev_sgg']

        # Task 6a: The Formula
        # Score = (0.5 * SGG) + (0.3 * Pose) + (0.2 * IoU)
        final_score = (self.W_SGG * current_sgg) + \
                      (self.W_POSE * pose_score) + \
                      (self.W_IOU * iou_score)

        # Task 6b: State Machine (Hysteresis)
        # We use dual thresholds to prevent flickering
        current_state = self.chair_states[chair_id]['state']
        
        new_state = current_state
        if current_state == 'EMPTY' and final_score > 0.65:
            new_state = 'OCCUPIED'
        elif current_state == 'OCCUPIED' and final_score < 0.40:
            new_state = 'EMPTY'

        self.chair_states[chair_id]['score'] = final_score
        self.chair_states[chair_id]['state'] = new_state
        
        return new_state, final_score