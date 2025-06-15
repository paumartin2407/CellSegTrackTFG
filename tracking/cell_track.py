class CellTrack:
    """
    Class representing a single cell track across frames.
    """

    def __init__(self, label: int, start_frame: int, end_frame: int = None, parent_label: int = 0):
        """
        Initialize a new CellTrack instance.

        Args:
            label (int): Unique track label (positive 16-bit integer).
            start_frame (int): Index of the starting frame (zero-based).
            end_frame (int, optional): Index of the ending frame (zero-based). Defaults to None.
            parent_label (int, optional): Label of the parent track (0 if no parent). Defaults to 0.
        """
        self.label = label
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.parent_label = parent_label
        self.positions = []  # List of (x, y) positions
        self.masks = []  # List of masks values
        self.missing_frames  = 0  

    def add_position(self, x: float, y: float):
        """
        Add a position to the cell track.

        Args:
            x (float): X coordinate.
            y (float): Y coordinate.
        """
        self.positions.append((x, y))

    def __repr__(self):
        """
        String representation of the CellTrack object.
        """
        return (f"CellTrack(label={self.label}, start_frame={self.start_frame}, "
                f"end_frame={self.end_frame}, parent_label={self.parent_label}, "
                f"positions={self.positions})")
