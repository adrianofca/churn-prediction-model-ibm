"""
mlp.py
------
Definição do MLP para classificação de churn em PyTorch.

Decisões de arquitetura:
  - 3 camadas escondidas: [128 → 64 → 32] → 1
  - Ativação: ReLU (padrão para tabular, sem vanishing gradient)
  - Regularização: BatchNorm + Dropout por camada
  - Saída: logit único (BCEWithLogitsLoss — numericamente estável)
  - Loss: BCEWithLogitsLoss com pos_weight (desbalanceamento 26.5%)
  - Inicialização: Kaiming uniform (recomendada para ReLU)

Todos os hiperparâmetros são configuráveis via MLPConfig / TrainConfig.
Para treino e avaliação, use trainer.py. Para persistência, use persistence.py.
"""

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ── Configuração da arquitetura ────────────────────────────────────────────────

@dataclass
class MLPConfig:
    """
    Hiperparâmetros do MLP — centralizados para facilitar tuning e MLflow.

    Atributos
    ---------
    input_dim       : número de features de entrada; sobrescrito automaticamente
                      pelo run_pipeline a partir do dataset carregado
    hidden_dims     : lista com o nº de neurônios por camada escondida
    dropout_rate    : probabilidade de dropout (aplicado após cada camada)
    use_batch_norm  : ativa BatchNormalization antes da ativação
    activation      : função de ativação ('relu', 'leaky_relu', 'elu')
    pos_weight      : peso da classe positiva na BCEWithLogitsLoss;
                      sobrescrito automaticamente pelo run_pipeline como
                      n_negativos / n_positivos do dataset carregado
    """
    input_dim:      int       = 30
    hidden_dims:    list[int] = field(default_factory=lambda: [128, 64])
    dropout_rate:   float     = 0.2
    use_batch_norm: bool      = True
    activation:     str       = "relu"
    pos_weight:     float     = 2.77   # n_neg / n_pos do dataset

    def to_dict(self) -> dict:
        """Serializa para log_params do MLflow."""
        return {
            "input_dim":      self.input_dim,
            "hidden_dims":    str(self.hidden_dims),
            "dropout_rate":   self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "activation":     self.activation,
            "pos_weight":     self.pos_weight,
        }


# ── Configuração de treino ─────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """
    Hiperparâmetros do training loop.

    Atributos
    ---------
    data_path     : URL ou caminho local do CSV bruto (dados originais)
    target_col    : nome da coluna alvo
    test_size     : fração para o conjunto de teste
    val_size      : fração para o conjunto de validação (relativa ao total)
    batch_size    : tamanho do batch
    epochs        : número máximo de épocas de treino
    learning_rate : taxa de aprendizado do otimizador Adam
    weight_decay  : L2 regularization no otimizador
    seed          : semente para reprodutibilidade
    device        : 'cuda' se disponível, senão 'cpu'
    patience      : épocas sem melhora antes de parar (0 = desativado)
    min_delta     : melhora mínima para ser considerada progresso
    monitor       : métrica monitorada ('val_loss', 'val_auc', 'val_f1' ou 'val_pr_auc')
    """
    data_path:     str   = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    target_col:    str   = "target"
    id_col:        str   = "customerID"
    test_size:     float = 0.15
    val_size:      float = 0.15
    batch_size:    int   = 64
    epochs:        int   = 100
    learning_rate: float = 1e-3
    weight_decay:  float = 1e-5
    seed:          int   = 42
    device:        str   = "cuda" if torch.cuda.is_available() else "cpu"
    patience:      int   = 10
    min_delta:     float = 1e-4
    monitor:       str   = "val_loss"

    def to_dict(self) -> dict:
        return {
            "batch_size":    self.batch_size,
            "epochs":        self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay":  self.weight_decay,
            "seed":          self.seed,
            "patience":      self.patience,
            "min_delta":     self.min_delta,
            "monitor":       self.monitor,
        }


# ── Bloco de camada escondida ──────────────────────────────────────────────────

class HiddenBlock(nn.Module):
    """
    Bloco reutilizável: Linear → [BatchNorm] → Activation → Dropout

    BatchNorm antes da ativação estabiliza a distribuição dos pré-ativações.
    """

    ACTIVATIONS = {
        "relu":       nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu":        nn.ELU,
    }

    def __init__(
        self,
        in_features:    int,
        out_features:   int,
        dropout_rate:   float,
        use_batch_norm: bool,
        activation:     str,
    ):
        super().__init__()

        if activation not in self.ACTIVATIONS:
            raise ValueError(
                f"Ativação '{activation}' inválida. "
                f"Escolha entre: {list(self.ACTIVATIONS.keys())}"
            )

        layers: list[nn.Module] = [nn.Linear(in_features, out_features)]

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_features))

        layers.append(self.ACTIVATIONS[activation]())
        layers.append(nn.Dropout(p=dropout_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Modelo principal ───────────────────────────────────────────────────────────

class ChurnMLP(nn.Module):
    """
    MLP para classificação binária de churn.

    Arquitetura padrão (MLPConfig default):
        Input(30) → Hidden(128) → Hidden(64) → Hidden(32) → Output(1)

    A camada de saída produz logits (sem Sigmoid) para uso com
    BCEWithLogitsLoss. Para probabilidades, use predict_proba().
    """

    def __init__(self, config: MLPConfig = MLPConfig()):
        super().__init__()
        self.config = config

        hidden_blocks = []
        in_dim = config.input_dim

        for out_dim in config.hidden_dims:
            hidden_blocks.append(
                HiddenBlock(
                    in_features=in_dim,
                    out_features=out_dim,
                    dropout_rate=config.dropout_rate,
                    use_batch_norm=config.use_batch_norm,
                    activation=config.activation,
                )
            )
            in_dim = out_dim

        self.hidden = nn.Sequential(*hidden_blocks)
        self.output = nn.Linear(in_dim, 1)
        self._init_weights()

    def _init_weights(self):
        """Inicialização Kaiming para convergência estável com ReLU."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — retorna logits (não probabilidades)."""
        x = self.hidden(x)
        return self.output(x).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna probabilidade P(churn=1) via Sigmoid dos logits."""
        training = self.training
        self.eval()
        with torch.no_grad():
            result = torch.sigmoid(self.forward(x))
        self.train(training)
        return result

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Retorna rótulos binários (0/1) dado um threshold de decisão."""
        return (self.predict_proba(x) >= threshold).long()

    def count_parameters(self) -> int:
        """Total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Loga resumo da arquitetura."""
        sep = "─" * 55
        logging.info(f"\n{sep}")
        logging.info("  ChurnMLP — Arquitetura")
        logging.info(sep)
        logging.info(f"  Input dim     : {self.config.input_dim}")
        logging.info(f"  Hidden layers : {self.config.hidden_dims}")
        logging.info(f"  Dropout       : {self.config.dropout_rate}")
        logging.info(f"  BatchNorm     : {self.config.use_batch_norm}")
        logging.info(f"  Activation    : {self.config.activation}")
        logging.info("  Output        : 1 (logit → BCEWithLogitsLoss)")
        logging.info(f"  pos_weight    : {self.config.pos_weight} (n_neg/n_pos)")
        logging.info(f"  Parâmetros    : {self.count_parameters():,}")
        logging.info(sep)


# ── Loss function ──────────────────────────────────────────────────────────────

def build_loss(config: MLPConfig, device: str = "cpu") -> nn.BCEWithLogitsLoss:
    """
    Constrói BCEWithLogitsLoss com pos_weight para compensar
    o desbalanceamento da classe positiva (churn = 26.5%).

    pos_weight = n_negativos / n_positivos ≈ 5174 / 1869 ≈ 2.77
    """
    pos_weight = torch.tensor([config.pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

