from typing import List, Sequence, Tuple

import torch


def text_pad(text: Sequence[str], period: int) -> List[str]:
    """Right-pad or trim a list of strings to a fixed length.

    Args:
        text: Sequence of strings.
        period: Target length.

    Returns:
        A list of length ``period``. If ``text`` is longer it is trimmed
        to the last ``period`` items, if shorter it is padded with "".
    """
    if len(text) > period:
        return list(text[-period:])
    if len(text) == period:
        return list(text)
    # pad with empty strings
    return list(text) + [""] * (period - len(text))


def created_pad(created: Sequence[int], period: int) -> torch.Tensor:
    """Normalize and pad/trim created-time indices.

    When longer than ``period``, keep the most recent ``period`` items and
    normalize so that the first element becomes 0. When shorter, right-pad
    with zeros. Values are returned as ``torch.long``.

    Args:
        created: Sequence of integer time indices.
        period: Target length.

    Returns:
        Tensor of shape (period,) with dtype long.
    """
    data = torch.tensor(created, dtype=torch.long)
    if len(created) > period:
        data = data[-period:]
        first_period = data[0].clone()
        data = data - first_period
        return data
    if len(created) == period:
        return data
    pad = torch.cat((data, torch.zeros(period - len(created), dtype=torch.long)))
    return pad


def make_pad(text: Sequence[str], period: int) -> torch.Tensor:
    """Create a padding mask where 1 indicates a valid position.

    Args:
        text: Sequence whose length determines the number of valid items.
        period: Target length.

    Returns:
        Float tensor of shape (period,) with ones for valid entries and zeros
        for padded positions.
    """
    if len(text) >= period:
        return torch.ones(period)
    return torch.cat((torch.ones(len(text)), torch.zeros(period - len(text))))


class to_padding:
    """Utilities to pad parallel sequences for model input.

    Note:
        Class name and public methods are preserved for backward compatibility.
    """

    def __init__(
        self,
        text: Sequence[Sequence[str]],
        created: Sequence[Sequence[int]],
        period: int,
    ) -> None:
        self.text = text
        self.created = created
        self.period = period

    def pad_data(self) -> Tuple[List[List[str]], torch.Tensor, torch.Tensor]:
        """Pad text, created indices and build padding mask.

        Returns:
            Tuple of (padded_text, padded_created, padding_mask)
        """
        text = self.pad_text()
        created = self.pad_created()
        padding = self.make_padding()
        return text, created, padding

    def pad_text(self) -> List[List[str]]:
        return [text_pad(i, self.period) for i in self.text]

    def pad_created(self) -> torch.Tensor:
        return torch.stack([created_pad(i, self.period) for i in self.created])

    def make_padding(self) -> torch.Tensor:
        return torch.stack([make_pad(i, self.period) for i in self.text])
