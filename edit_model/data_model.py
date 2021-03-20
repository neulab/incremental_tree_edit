from edit_model import nn_utils
from edit_model.utils import cached_property


class BatchedCodeChunk:
    def __init__(self, code_list, vocab, device=None, append_boundary_sym=False):
        self.code_list = code_list

        self.max_len = max(len(code) for code in code_list)
        self.batch_size = len(code_list)

        self._vocab = vocab
        self._append_boundary_sym = append_boundary_sym
        self.device=device

        # to be set by encoders
        self.encoding = None
        self.last_state = None
        self.last_cell = None

    @cached_property
    def index_var(self):
        return nn_utils.to_input_variable(self.code_list,
                                          vocab=self._vocab,
                                          device=self.device,
                                          append_boundary_sym=self._append_boundary_sym)

    @cached_property
    def mask(self):
        """mask for attention, null entries are masked as **1** """

        assert self._append_boundary_sym is False

        return nn_utils.length_array_to_mask_tensor([len(x) for x in self.code_list], device=self.device)
