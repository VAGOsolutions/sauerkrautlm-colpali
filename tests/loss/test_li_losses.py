# ruff: noqa: N806, N812
import pytest
import torch
import torch.nn.functional as F

from sauerkrautlm_colpali.loss import (
    ColbertDistillationLoss,
    ColbertLoss,
    ColbertModule,
    ColbertNegativeCELoss,
    ColbertPairwiseCELoss,
    ColbertPairwiseNegativeCELoss,
)


class TestColbertModule:
    def test_get_idx(self):
        module = ColbertModule(max_batch_size=5)
        idx, pos_idx = module._get_idx(batch_size=3, offset=2, device=torch.device("cpu"))
        assert torch.equal(idx, torch.tensor([0, 1, 2]))
        assert torch.equal(pos_idx, torch.tensor([2, 3, 4]))

    def test_smooth_max(self):
        module = ColbertModule(tau=2.0)
        scores = torch.tensor([[0.0, 2.0]])
        out = module._smooth_max(scores, dim=1)
        expected = 2.0 * torch.log(torch.tensor(1.0 + torch.exp(torch.tensor(1.0))))
        assert torch.allclose(out, expected)

    def test_apply_normalization_within_bounds(self):
        module = ColbertModule(norm_tol=1e-3)
        scores = torch.tensor([[0.5, 1.0], [0.2, 0.8]])
        lengths = torch.tensor([2.0, 4.0])
        normalized = module._apply_normalization(scores, lengths)
        expected = scores / lengths.unsqueeze(1)
        assert torch.allclose(normalized, expected)

    # def test_apply_normalization_out_of_bounds(self):
    #     module = ColbertModule(norm_tol=1e-3)
    #     scores = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
    #     lengths = torch.tensor([1.0, 1.0])
    #     with pytest.raises(ValueError) as excinfo:
    #         module._apply_normalization(scores, lengths)
    #     assert "Scores out of bounds after normalization" in str(excinfo.value)

    def test_aggregate_max(self):
        module = ColbertModule()
        raw = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        out = module._aggregate(raw, use_smooth_max=False, dim_max=2, dim_sum=1)
        assert torch.allclose(out, torch.tensor([6.0, 14.0]))

    def test_aggregate_smooth_max(self):
        module = ColbertModule(tau=1.0)
        raw = torch.zeros(1, 2, 2)
        out = module._aggregate(raw, use_smooth_max=True, dim_max=2, dim_sum=1)
        assert torch.allclose(out, 2 * torch.log(torch.tensor(2.0)))

    def test_filter_high_negatives(self):
        module = ColbertModule(filter_threshold=0.95, filter_factor=0.5)
        scores = torch.tensor([[1.0, 0.96], [0.5, 1.0]])
        original = scores.clone()
        pos_idx = torch.tensor([0, 1])
        module._filter_high_negatives(scores, pos_idx)
        assert scores[0, 1] == pytest.approx(0.48)
        # other entries unchanged
        assert scores[0, 0] == original[0, 0]
        assert scores[1, 0] == original[1, 0]
        assert scores[1, 1] == original[1, 1]


class TestColbertLoss:
    def test_zero_embeddings(self):
        loss_fn = ColbertLoss(
            temperature=1.0,
            normalize_scores=False,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
        )
        B, Nq, D = 3, 1, 4
        query = torch.zeros(B, Nq, D)
        doc = torch.zeros(B, Nq, D)
        loss = loss_fn(query, doc)
        expected = torch.log(torch.tensor(float(B)))
        assert torch.allclose(loss, expected)

    def test_with_and_without_filtering(self):
        base = ColbertLoss(
            temperature=1.0, normalize_scores=False, use_smooth_max=False, pos_aware_negative_filtering=False
        )
        filt = ColbertLoss(
            temperature=1.0, normalize_scores=False, use_smooth_max=False, pos_aware_negative_filtering=True
        )
        B, Nq, D = 2, 1, 3
        query = torch.zeros(B, Nq, D)
        doc = torch.zeros(B, Nq, D)
        assert torch.allclose(base(query, doc), filt(query, doc))


class TestColbertNegativeCELoss:
    def test_no_inbatch(self):
        loss_fn = ColbertNegativeCELoss(
            temperature=1.0,
            normalize_scores=False,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
            in_batch_term_weight=0,
        )
        B, Nq, D, Nneg = 2, 1, 3, 1
        query = torch.zeros(B, Nq, D)
        doc = torch.zeros(B, Nq, D)
        neg = torch.zeros(B, Nneg, D)
        loss = loss_fn(query, doc, neg)
        expected = F.softplus(torch.tensor(0.0))
        assert torch.allclose(loss, expected)

    def test_with_inbatch(self):
        loss_fn = ColbertNegativeCELoss(
            temperature=1.0,
            normalize_scores=False,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
            in_batch_term_weight=0.5,
        )
        B, Nq, D = 2, 1, 3
        query = torch.zeros(B, Nq, D)
        doc = torch.zeros(B, Nq, D)
        neg = torch.zeros(B, 1, D)
        loss = loss_fn(query, doc, neg)
        expected = F.softplus(torch.tensor(0.0))
        assert torch.allclose(loss, expected)


class TestColbertPairwiseCELoss:
    def test_zero_embeddings(self):
        loss_fn = ColbertPairwiseCELoss(
            temperature=1.0, normalize_scores=False, use_smooth_max=False, pos_aware_negative_filtering=False
        )
        B, Nq, D = 2, 1, 3
        query = torch.zeros(B, Nq, D)
        doc = torch.zeros(B, Nq, D)
        loss = loss_fn(query, doc)
        expected = F.softplus(torch.tensor(0.0))
        assert torch.allclose(loss, expected)


class TestColbertPairwiseNegativeCELoss:
    def test_no_inbatch(self):
        loss_fn = ColbertPairwiseNegativeCELoss(
            temperature=1.0,
            normalize_scores=False,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
            in_batch_term_weight=0,
        )
        B, Nq, D = 2, 1, 3
        query = torch.zeros(B, Nq, D)
        doc = torch.zeros(B, Nq, D)
        neg = torch.zeros(B, 1, D)
        loss = loss_fn(query, doc, neg)
        expected = F.softplus(torch.tensor(0.0))
        assert torch.allclose(loss, expected)

    def test_with_inbatch(self):
        loss_fn = ColbertPairwiseNegativeCELoss(
            temperature=1.0,
            normalize_scores=False,
            use_smooth_max=False,
            pos_aware_negative_filtering=False,
            in_batch_term_weight=0.5,
        )
        B, Nq, D = 2, 1, 3
        query = torch.zeros(B, Nq, D)
        doc = torch.zeros(B, Nq, D)
        neg = torch.zeros(B, 1, D)
        loss = loss_fn(query, doc, neg)
        expected = F.softplus(torch.tensor(0.0))
        assert torch.allclose(loss, expected)


class TestColbertDistillationLoss:
    def test_basic_functionality(self):
        """Test basic functionality of the distillation loss."""
        loss_fn = ColbertDistillationLoss(
            temperature=1.0,
            normalize_scores=False,
            use_smooth_max=False,
            score_normalization="none",
        )

        B, Nq, N_docs, Nd, D = 2, 3, 4, 5, 8
        query_embeddings = torch.randn(B, Nq, D)
        doc_embeddings = torch.randn(B, N_docs, Nd, D)
        teacher_scores = torch.randn(B, N_docs)

        loss = loss_fn(query_embeddings, doc_embeddings, teacher_scores)

        # Loss should be a scalar tensor
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0  # KL divergence is non-negative

    def test_score_normalization(self):
        """Test min-max score normalization."""
        loss_fn = ColbertDistillationLoss(
            temperature=1.0,
            normalize_scores=False,
            use_smooth_max=False,
            score_normalization="minmax",
        )

        B, Nq, N_docs, Nd, D = 1, 2, 3, 4, 8
        query_embeddings = torch.randn(B, Nq, D)
        doc_embeddings = torch.randn(B, N_docs, Nd, D)
        teacher_scores = torch.randn(B, N_docs)

        loss = loss_fn(query_embeddings, doc_embeddings, teacher_scores)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_smooth_max_aggregation(self):
        """Test smooth max aggregation instead of regular max."""
        loss_fn = ColbertDistillationLoss(
            temperature=1.0,
            normalize_scores=False,
            use_smooth_max=True,
            score_normalization="none",
            tau=0.1,
        )

        B, Nq, N_docs, Nd, D = 1, 2, 2, 3, 4
        query_embeddings = torch.randn(B, Nq, D)
        doc_embeddings = torch.randn(B, N_docs, Nd, D)
        teacher_scores = torch.randn(B, N_docs)

        loss = loss_fn(query_embeddings, doc_embeddings, teacher_scores)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_query_length_normalization(self):
        """Test normalization by query lengths."""
        loss_fn = ColbertDistillationLoss(
            temperature=1.0,
            normalize_scores=True,
            use_smooth_max=False,
            score_normalization="none",
        )

        B, Nq, N_docs, Nd, D = 1, 3, 2, 2, 4
        query_embeddings = torch.randn(B, Nq, D)
        # Set some query tokens to zero to simulate padding
        query_embeddings[0, 2, 0] = 0  # This token should be padding

        doc_embeddings = torch.randn(B, N_docs, Nd, D)
        teacher_scores = torch.randn(B, N_docs)

        loss = loss_fn(query_embeddings, doc_embeddings, teacher_scores)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_identical_scores_zero_loss(self):
        """Test that identical student and teacher scores result in zero loss."""
        loss_fn = ColbertDistillationLoss(
            temperature=1.0,
            normalize_scores=False,
            use_smooth_max=False,
            score_normalization="none",
        )

        # Create deterministic embeddings that will produce predictable scores
        B, Nq, N_docs, Nd, D = 1, 1, 2, 1, 2
        query_embeddings = torch.ones(B, Nq, D)
        doc_embeddings = torch.ones(B, N_docs, Nd, D)

        # Since all embeddings are identical, scores will be identical
        # We need to provide teacher scores that match this
        with torch.no_grad():
            student_scores = loss_fn._compute_colbert_scores(
                F.normalize(query_embeddings, p=2, dim=-1),
                F.normalize(doc_embeddings, p=2, dim=-1)
            )

        teacher_scores = student_scores.clone()

        loss = loss_fn(query_embeddings, doc_embeddings, teacher_scores)

        # Should be very close to zero (within numerical precision)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_temperature_scaling(self):
        """Test that temperature scaling affects the loss."""
        B, Nq, N_docs, Nd, D = 1, 2, 2, 2, 4
        query_embeddings = torch.randn(B, Nq, D)
        doc_embeddings = torch.randn(B, N_docs, Nd, D)
        teacher_scores = torch.randn(B, N_docs)

        loss_fn_low_temp = ColbertDistillationLoss(temperature=0.1)
        loss_fn_high_temp = ColbertDistillationLoss(temperature=10.0)

        loss_low = loss_fn_low_temp(
            query_embeddings, doc_embeddings, teacher_scores
        )
        loss_high = loss_fn_high_temp(
            query_embeddings, doc_embeddings, teacher_scores
        )

        # Both should be valid losses
        assert loss_low.item() >= 0
        assert loss_high.item() >= 0

        # They should be different due to temperature scaling
        assert not torch.allclose(loss_low, loss_high, atol=1e-6)
