"""CR-BSRO forecasting model."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def moving_average(x: torch.Tensor, kernel: int) -> torch.Tensor:
    padding = (kernel - 1) // 2
    padded = F.pad(x.transpose(1, 2), (padding, padding), mode="replicate")
    return F.avg_pool1d(padded, kernel, stride=1).transpose(1, 2)


def initialize_last_value(layer: nn.Linear) -> None:
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    with torch.no_grad():
        layer.weight[:, -1] = 1.0


class FactorizedTemporalOperator(nn.Module):
    def __init__(self, history: int, horizon: int, rank: int):
        super().__init__()
        if rank <= 0:
            raise ValueError("temporal rank must be positive")
        self.analysis = nn.Linear(history, rank, bias=False)
        self.synthesis = nn.Linear(rank, horizon)
        nn.init.zeros_(self.synthesis.weight)
        nn.init.zeros_(self.synthesis.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.synthesis(self.analysis(x))


class TokenBlock(nn.Module):
    def __init__(self, width: int, expansion: float, dropout: float):
        super().__init__()
        inner = max(width, int(width * expansion))
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(
            nn.Linear(width, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class InducedChannelOperator(nn.Module):
    """Low-rank channel communication through a fixed number of slots."""

    def __init__(self, width: int, slots: int, dropout: float,
                 scale_init: float):
        super().__init__()
        if slots <= 0:
            raise ValueError("channel slots must be positive")
        self.pool = nn.Linear(width, slots, bias=False)
        self.read = nn.Linear(width, slots, bias=False)
        self.value = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(width * 2, width)
        self.scale_logit = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pool = self.pool(tokens).softmax(dim=1)
        slots = torch.einsum("bck,bcd->bkd", pool, tokens)
        read = self.read(tokens).softmax(dim=-1)
        context = self.value(torch.einsum("bck,bkd->bcd", read, slots))
        gate = torch.sigmoid(self.gate(torch.cat((tokens, context), dim=-1)))
        return tokens + torch.sigmoid(self.scale_logit) * gate * context


class CausalPhaseProjector(nn.Module):
    """Estimate and extrapolate a low-rank phase signal causally."""

    def __init__(
        self,
        period: int,
        channels: int,
        rank: int,
        ridge: float,
        prior_strength: float,
        scale_init: float,
        update_scale_init: float,
        residual_scale_init: float,
        basis_init: Optional[torch.Tensor] = None,
        coefficient_init: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if period <= 0:
            raise ValueError("phase period must be positive")
        if rank <= 0:
            raise ValueError("phase rank must be positive")

        self.period = int(period)
        self.rank = int(rank)
        self.phase_table = nn.Embedding(self.period, self.rank)

        phase = torch.arange(self.period, dtype=torch.float32)[:, None]
        order = torch.arange(
            1, (self.rank + 1) // 2 + 1, dtype=torch.float32
        )[None, :]
        fourier = torch.cat(
            (
                torch.sin(2 * math.pi * phase * order / self.period),
                torch.cos(2 * math.pi * phase * order / self.period),
            ),
            dim=1,
        )
        with torch.no_grad():
            if basis_init is None:
                self.phase_table.weight.copy_(fourier[:, : self.rank])
            else:
                if tuple(basis_init.shape) != (self.period, self.rank):
                    raise ValueError("phase basis has an incompatible shape")
                self.phase_table.weight.copy_(basis_init)

        self.coefficient_prior = nn.Parameter(torch.zeros(channels, self.rank))
        if coefficient_init is not None:
            if tuple(coefficient_init.shape) != (channels, self.rank):
                raise ValueError("phase coefficients have an incompatible shape")
            with torch.no_grad():
                self.coefficient_prior.copy_(coefficient_init)

        self.ridge_log = nn.Parameter(torch.tensor(math.log(float(ridge))))
        self.prior_log = nn.Parameter(
            torch.tensor(math.log(float(prior_strength)))
        )
        self.scale_logit = nn.Parameter(torch.tensor(float(scale_init)))
        self.update_scale_logit = nn.Parameter(
            torch.tensor(float(update_scale_init))
        )
        self.residual_scale_logit = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )

    def forward(
        self,
        history: torch.Tensor,
        history_mark: torch.Tensor,
        future_mark: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The final marker is a normalized phase coordinate.
        history_index = torch.round(history_mark[..., -1] * self.period).long()
        future_index = torch.round(future_mark[..., -1] * self.period).long()
        history_basis = self.phase_table(
            history_index.remainder(self.period)
        )
        future_basis = self.phase_table(
            future_index.remainder(self.period)
        )

        eye = torch.eye(
            self.rank, device=history.device, dtype=history.dtype
        )
        ridge = self.ridge_log.exp().clamp(1e-4, 1e2)
        prior_strength = self.prior_log.exp().clamp(1e-4, 1e3)
        prior = self.coefficient_prior.unsqueeze(0).expand(
            history.size(0), -1, -1
        )

        # Fit on a prefix and score on a held-out suffix before using the
        # phase signal for the future.
        split = max(self.rank + 1, 2 * history.size(1) // 3)
        if split >= history.size(1):
            raise ValueError("sequence length is too short for phase cross-fit")
        fit_basis = history_basis[:, :split]
        fit_history = history[:, :split]
        fit_gram = torch.einsum("btr,bts->brs", fit_basis, fit_basis)
        fit_rhs = torch.einsum("btr,btc->brc", fit_basis, fit_history)
        fit_coefficients = torch.linalg.solve(
            fit_gram + (ridge + prior_strength) * eye.unsqueeze(0),
            fit_rhs + prior_strength * prior.transpose(1, 2),
        )
        held_prediction = torch.einsum(
            "btr,brc->btc", history_basis[:, split:], fit_coefficients
        )
        held_target = history[:, split:]
        held_error = (held_target - held_prediction).square().mean(dim=1)
        zero_error = held_target.square().mean(dim=1).clamp_min(1e-5)
        skill = 1.0 - held_error / zero_error
        reliability = torch.sigmoid(6.0 * (skill - 0.25))

        gram = torch.einsum("btr,bts->brs", history_basis, history_basis)
        rhs = torch.einsum("btr,btc->brc", history_basis, history)
        coefficients = torch.linalg.solve(
            gram + (ridge + prior_strength) * eye.unsqueeze(0),
            rhs + prior_strength * prior.transpose(1, 2),
        )
        prior_forecast = torch.einsum(
            "bhr,brc->bhc", future_basis, prior.transpose(1, 2)
        )
        posterior_forecast = torch.einsum(
            "bhr,brc->bhc", future_basis, coefficients
        )
        forecast = (
            torch.sigmoid(self.scale_logit) * prior_forecast
            + torch.sigmoid(self.update_scale_logit)
            * reliability.unsqueeze(1)
            * (posterior_forecast - prior_forecast)
        )
        history_prior = torch.sigmoid(self.residual_scale_logit) * torch.einsum(
            "btr,brc->btc", history_basis, prior.transpose(1, 2)
        )
        return forecast, reliability, history_prior


class CRBSRO(nn.Module):
    """The full forecasting architecture."""

    def __init__(self, config):
        super().__init__()
        self.architecture_id = "CR-BSRO"
        self.seq_len = int(config.seq_len)
        self.pred_len = int(config.pred_len)
        channels = int(config.enc_in)

        width = int(getattr(config, "hidden", 32))
        layers = int(getattr(config, "layers", 1))
        temporal_rank = int(getattr(config, "temporal_rank", 16))
        phase_rank = int(getattr(config, "phase_rank", 8))
        channel_rank = int(getattr(config, "channel_rank", 8))
        channel_slots = int(getattr(config, "channel_slots", 8))
        kernel = int(getattr(config, "kernel", 25))
        dropout = float(getattr(config, "dropout", 0.1))
        expansion = float(getattr(config, "expansion", 1.5))
        if kernel <= 0 or kernel % 2 != 1:
            raise ValueError("kernel must be a positive odd integer")
        if layers <= 0:
            raise ValueError("layers must be positive")
        self.kernel = kernel

        self.trend_root = nn.Linear(self.seq_len, self.pred_len)
        self.season_root = nn.Linear(self.seq_len, self.pred_len)
        initialize_last_value(self.trend_root)
        initialize_last_value(self.season_root)
        self.delta_operator = FactorizedTemporalOperator(
            self.seq_len - 1, self.pred_len, temporal_rank
        )

        self.trend_read = nn.Linear(self.seq_len, width)
        self.season_read = nn.Linear(self.seq_len, width)
        self.delta_read = nn.Linear(self.seq_len, width)
        self.read_norm = nn.LayerNorm(width)
        self.channel_code = nn.Parameter(torch.empty(channels, channel_rank))
        self.channel_basis = nn.Linear(channel_rank, width, bias=False)
        nn.init.normal_(self.channel_code, std=channel_rank ** -0.5)
        self.spectral_film = nn.Linear(4, width * 2)
        nn.init.zeros_(self.spectral_film.weight)
        nn.init.zeros_(self.spectral_film.bias)
        self.blocks = nn.ModuleList(
            TokenBlock(width, expansion, dropout) for _ in range(layers)
        )
        self.channel_operator = InducedChannelOperator(
            width,
            channel_slots,
            dropout,
            float(getattr(config, "channel_scale_init", -2.0)),
        )
        self.state_head = nn.Linear(width, self.pred_len)
        nn.init.zeros_(self.state_head.weight)
        nn.init.zeros_(self.state_head.bias)

        self.phase_operator = CausalPhaseProjector(
            int(config.phase_period),
            channels,
            phase_rank,
            float(getattr(config, "phase_ridge", 1.0)),
            float(getattr(config, "phase_prior_strength", 8.0)),
            float(getattr(config, "phase_scale_init", -2.0)),
            float(getattr(config, "phase_update_scale_init", -2.0)),
            float(getattr(config, "phase_residual_scale_init", 4.0)),
            getattr(config, "phase_basis_init", None),
            getattr(config, "phase_coefficient_init", None),
        )
        self.router = nn.Sequential(
            nn.Linear(7, max(8, width // 2)),
            nn.GELU(),
            nn.Linear(max(8, width // 2), 3),
        )
        nn.init.zeros_(self.router[-1].weight)
        with torch.no_grad():
            self.router[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0]))
        self.channel_route = nn.Parameter(torch.zeros(channels, 3))

        # This low-rank temporal adapter is part of the common backbone.
        factor_rank = int(getattr(config, "factor_rank", 1))
        if factor_rank <= 0:
            raise ValueError("factor rank must be positive")
        self.factor_basis = nn.Parameter(
            torch.zeros(factor_rank, self.pred_len, self.seq_len)
        )
        self.factor_coeff = nn.Parameter(torch.empty(channels, factor_rank))
        nn.init.normal_(self.factor_coeff, std=factor_rank ** -0.5)
        self.factor_scale_logit = nn.Parameter(
            torch.tensor(float(getattr(config, "factor_scale_init", -2.0)))
        )

        calendar_rank = int(getattr(config, "calendar_rank", 8))
        if calendar_rank <= 0:
            raise ValueError("calendar rank must be positive")
        self.calendar_rank = calendar_rank
        self.calendar_time = nn.Sequential(
            nn.LayerNorm(int(config.mark_dim)),
            nn.Linear(int(config.mark_dim), calendar_rank),
            nn.Tanh(),
        )
        self.calendar_sample = nn.Linear(4, calendar_rank, bias=False)
        self.calendar_channel = nn.Parameter(torch.empty(channels, calendar_rank))
        nn.init.zeros_(self.calendar_sample.weight)
        nn.init.normal_(self.calendar_channel, std=0.01)
        self.calendar_scale_logit = nn.Parameter(
            torch.tensor(float(getattr(config, "calendar_init", -2.0)))
        )
        self.calendar_decay_log = nn.Parameter(
            torch.tensor(float(getattr(config, "calendar_decay_init", 1.0)))
        )

    @staticmethod
    def descriptors(
        z: torch.Tensor, phase_reliability: torch.Tensor
    ) -> torch.Tensor:
        delta = z[:, 1:] - z[:, :-1]
        energy = torch.fft.rfft(z, dim=1).abs().square()
        bins = energy.size(1)
        cut1 = max(2, bins // 3)
        cut2 = max(cut1 + 1, 2 * bins // 3)
        total = energy[:, 1:].mean(1).clamp_min(1e-6)
        low = energy[:, 1:cut1].mean(1) / total
        high = energy[:, cut2:].mean(1) / total
        return torch.stack(
            (
                z[:, -1],
                z[:, -1] - z[:, 0],
                delta.abs().mean(1),
                delta.square().mean(1).sqrt(),
                low,
                high,
                phase_reliability,
            ),
            dim=-1,
        )

    @staticmethod
    def spectral_descriptors(z: torch.Tensor) -> torch.Tensor:
        energy = torch.fft.rfft(z, dim=1).abs().square()
        bins = energy.size(1)
        cut1 = max(2, bins // 3)
        cut2 = max(cut1 + 1, 2 * bins // 3)
        total = energy[:, 1:].mean(1)
        low = energy[:, 1:cut1].mean(1)
        mid = energy[:, cut1:cut2].mean(1)
        high = energy[:, cut2:].mean(1)
        return torch.stack((low, mid, high, total), dim=-1).clamp_min(0).log1p()

    @staticmethod
    def calendar_descriptor(z: torch.Tensor) -> torch.Tensor:
        delta = z[:, 1:] - z[:, :-1]
        return torch.stack(
            (
                z[:, -1],
                z[:, -1] - z[:, 0],
                delta.mean(dim=1),
                delta.square().mean(dim=1).sqrt(),
            ),
            dim=-1,
        )

    def _backbone_forecast(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor,
        x_mark_future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase, phase_reliability, history_phase = self.phase_operator(
            x, x_mark, x_mark_future
        )
        residual_x = x - history_phase
        means = residual_x.mean(1, keepdim=True).detach()
        centered = residual_x - means
        stdev = centered.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt()
        z = centered / stdev

        trend = moving_average(z, self.kernel)
        season = z - trend
        delta = z[:, 1:] - z[:, :-1]
        delta_padded = F.pad(delta.transpose(1, 2), (1, 0))
        trend_c = trend.transpose(1, 2)
        season_c = season.transpose(1, 2)
        z_c = z.transpose(1, 2)

        structural = self.trend_root(trend_c) + self.season_root(season_c)
        incremental = z_c[:, :, -1:] + self.delta_operator(delta.transpose(1, 2))

        tokens = self.read_norm(
            self.trend_read(trend_c)
            + self.season_read(season_c)
            + self.delta_read(delta_padded)
            + self.channel_basis(self.channel_code).unsqueeze(0)
        )
        gain, bias = self.spectral_film(self.spectral_descriptors(z)).chunk(2, -1)
        tokens = tokens * (1.0 + 0.1 * torch.tanh(gain)) + 0.1 * bias
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.channel_operator(tokens)
        state = z_c[:, :, -1:] + self.state_head(tokens)

        weights = (
            self.router(self.descriptors(z, phase_reliability))
            + self.channel_route.unsqueeze(0)
        ).softmax(dim=-1)
        local = (
            weights[..., 0:1] * structural
            + weights[..., 1:2] * incremental
            + weights[..., 2:3] * state
        )
        output = local.transpose(1, 2) * stdev + means + phase

        # Preserve the common low-rank adapter's use of the original input.
        input_mean = x.mean(dim=1, keepdim=True).detach()
        input_centered = x - input_mean
        input_std = input_centered.var(
            dim=1, keepdim=True, unbiased=False
        ).add(1e-5).sqrt()
        normalized = input_centered / input_std
        atoms = torch.einsum(
            "bct,rht->bcrh", normalized.transpose(1, 2), self.factor_basis
        )
        adapter = torch.einsum(
            "bcrh,cr->bch", atoms, self.factor_coeff
        ).transpose(1, 2)
        output = output + torch.sigmoid(self.factor_scale_logit) * adapter * input_std
        return output, z

    def _calendar_posterior(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor,
        x_mark_future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output, _ = self._backbone_forecast(x, x_mark, x_mark_future)
        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stdev = centered.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt()
        z = centered / stdev
        amplitude = self.calendar_channel.unsqueeze(0) + self.calendar_sample(
            self.calendar_descriptor(z)
        )
        time_basis = self.calendar_time(x_mark_future)
        calendar = torch.einsum(
            "bhr,bcr->bhc", time_basis, amplitude
        ) / math.sqrt(self.calendar_rank)
        output = output + torch.sigmoid(self.calendar_scale_logit) * calendar * stdev
        return output, calendar, stdev

    def _calendar_terms(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor,
        x_mark_future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output, _, _ = self._calendar_posterior(
            x, x_mark, x_mark_future
        )
        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stdev = centered.var(1, keepdim=True, unbiased=False).add(1e-5).sqrt()
        z = centered / stdev
        # Preserve the original v24 backward graph: its subtraction branch
        # recomputes the calendar instead of sharing the v19 addition node.
        # Sharing is forward-equivalent but changes float32 gradient sums.
        full_amplitude = self.calendar_channel.unsqueeze(0) + self.calendar_sample(
            self.calendar_descriptor(z)
        )
        split = 2 * self.seq_len // 3
        future_basis = self.calendar_time(x_mark_future)
        calendar = torch.einsum(
            "bhr,bcr->bhc", future_basis, full_amplitude
        ) / math.sqrt(self.calendar_rank)
        prefix = z[:, :split]
        prefix_amplitude = self.calendar_channel.unsqueeze(0) + self.calendar_sample(
            self.calendar_descriptor(prefix)
        )
        held_basis = self.calendar_time(x_mark[:, split:])
        held_calendar = torch.einsum(
            "btr,bcr->btc", held_basis, prefix_amplitude
        ) / math.sqrt(self.calendar_rank)
        held_target = z[:, split:]
        calendar_error = (held_target - held_calendar).square().mean(dim=1)
        zero_error = held_target.square().mean(dim=1).clamp_min(1e-5)
        skill = 1.0 - calendar_error / zero_error
        reliability = torch.sigmoid(6.0 * (skill - 0.1))
        gate = torch.sigmoid(self.calendar_scale_logit)
        return output, calendar, stdev, reliability, gate

    def _lead_decay(self) -> torch.Tensor:
        lead = torch.arange(
            1,
            self.pred_len + 1,
            device=self.calendar_decay_log.device,
            dtype=self.calendar_decay_log.dtype,
        )
        decay_rate = F.softplus(self.calendar_decay_log)
        return torch.exp(-decay_rate * lead / self.pred_len)

    def forecast(
        self,
        x: torch.Tensor,
        x_mark: Optional[torch.Tensor] = None,
        x_mark_future: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_mark is None or x_mark_future is None:
            raise ValueError("CR-BSRO requires historical and future markers")

        output, calendar, stdev, reliability, gate = self._calendar_terms(
            x, x_mark, x_mark_future
        )
        effective = reliability.unsqueeze(1) * self._lead_decay().view(1, -1, 1)
        return output - gate * (1.0 - effective) * calendar * stdev

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forecast(x_enc, x_mark_enc, x_mark_dec)


def build_model(config) -> CRBSRO:
    return CRBSRO(config)
