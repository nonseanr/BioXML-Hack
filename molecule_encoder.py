import torch

from tqdm import tqdm
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer


class SmallMoleculeEncoder(torch.nn.Module):
    def __init__(self,
                 config_path: str,
                 out_dim: int,
                 load_pretrained: bool = True,
                 gradient_checkpointing: bool = False):
        """
        Args:
            config_path: Path to the config file
            
            out_dim    : Output dimension of the protein representation
            
            load_pretrained: Whether to load pretrained weights
            
            gradient_checkpointing: Whether to use gradient checkpointing
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(config_path, deterministic_eval=True, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(config_path, trust_remote_code=True)

        # TODO: remove
        for param in self.model.parameters():
            param.requires_grad = False

        # config = EsmConfig.from_pretrained(config_path)
        # if load_pretrained:
        #     self.model = EsmForMaskedLM.from_pretrained(config_path)
        # else:
        #     self.model = EsmForMaskedLM(config)
        self.out = torch.nn.Linear(self.model.config.hidden_size, out_dim)
        
        # # Set gradient checkpointing
        # self.model.esm.encoder.gradient_checkpointing = gradient_checkpointing
        
        # # Remove contact head
        # self.model.esm.contact_head = None
        
        # # Remove position embedding if the embedding type is ``rotary``
        # if config.position_embedding_type == "rotary":
        #     self.model.esm.embeddings.position_embeddings = None
        
        # self.tokenizer = EsmTokenizer.from_pretrained(config_path)
    
    def get_repr(self, molecules: list, batch_size: int = 64, verbose: bool = False) -> torch.Tensor:
        """
        Compute protein representation for the given proteins
        Args:
            protein: A list of protein sequences
            batch_size: Batch size for inference
            verbose: Whether to print progress
        """
        device = next(self.parameters()).device
        
        sm_repr = []
        if verbose:
            iterator = tqdm(range(0, len(molecules), batch_size), desc="Computing SM embeddings")
        else:
            iterator = range(0, len(molecules), batch_size)
            
        for i in iterator:
            sm_inputs = self.tokenizer.batch_encode_plus(molecules[i:i + batch_size],
                                                              return_tensors="pt",
                                                              padding=True)
            sm_inputs = {k: v.to(device) for k, v in sm_inputs.items()}
            output = self.forward(sm_inputs)
            
            sm_repr.append(output)
        
        sm_repr = torch.cat(sm_repr, dim=0)
        return normalize(sm_repr, dim=-1)

    def forward(self, inputs: dict):
        """
        Encode protein sequence into protein representation
        Args:
            inputs: A dictionary containing the following keys:
                - input_ids: [batch, seq_len]
                - attention_mask: [batch, seq_len]

        Returns:
            protein_repr: [batch, protein_repr_dim]
        """
        last_hidden_state = self.model(**inputs).last_hidden_state
        reprs = last_hidden_state[:, 0, :]
        return self.out(reprs)