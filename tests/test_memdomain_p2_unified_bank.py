import unittest

from scalesim.memory.memdomain_adapter import policy_result_from_report
from scalesim.memory.memdomain_policy import ResourceBudget
from scalesim.memory.unified_bank_domain import UnifiedBankDomain, UnifiedMemoryRequest


class UnifiedBankDomainTests(unittest.TestCase):
    def setUp(self):
        self.resources = ResourceBudget(4, 4096, 64, 1, 8)
        self.domain = UnifiedBankDomain(self.resources, interleave_bytes=16)

    @staticmethod
    def request(request_id, tensor, address=0, issue=0, kind="read", size=16):
        return UnifiedMemoryRequest(request_id, issue, tensor, request_id, address, size, kind)

    def test_all_tensor_types_share_one_physical_bank_namespace(self):
        requests = [
            self.request("ia", "ia", 0),
            self.request("w", "weight", 0),
            self.request("oa", "oa", 0, kind="write"),
            self.request("acc", "accumulator", 0, kind="write"),
        ]
        report = self.domain.simulate(requests)
        self.assertEqual(report.per_bank_accesses[0], 4)
        self.assertEqual(report.per_bank_conflicts[0], 3)
        self.assertEqual(sum(report.per_tensor_requests.values()), 4)

    def test_prefetch_and_compute_contend_on_same_bank(self):
        compute = self.request("compute", "weight", 0, kind="read")
        prefetch = self.request("prefetch", "weight", 0, kind="prefetch")
        report = self.domain.simulate([prefetch, compute])
        self.assertGreater(report.total_queue_wait_cycles, 0)
        self.assertEqual(report.per_bank_conflicts[0], 1)
        self.assertEqual(report.services[0].request_id, "compute")

    def test_interleaving_uses_multiple_physical_banks(self):
        report = self.domain.simulate([self.request("large", "weight", 0, size=64)])
        self.assertEqual(report.services[0].banks, (0, 1, 2, 3))
        self.assertEqual(report.total_beats, 4)

    def test_total_bytes_and_beats_are_conserved(self):
        requests = [self.request("a", "ia", 7, size=31), self.request("b", "oa", 33, size=19)]
        report = self.domain.simulate(requests)
        self.assertEqual(report.total_bytes, 50)
        self.assertEqual(sum(report.per_bank_accesses.values()), report.total_beats)

    def test_additional_port_removes_same_cycle_two_request_conflict(self):
        resources = ResourceBudget(4, 4096, 64, 2, 8)
        domain = UnifiedBankDomain(resources, interleave_bytes=16)
        report = domain.simulate([
            self.request("a", "ia", 0), self.request("b", "weight", 0)
        ])
        self.assertEqual(report.per_bank_conflicts[0], 0)

    def test_request_buffer_depth_bounds_outstanding_queue(self):
        resources = ResourceBudget(1, 4096, 16, 1, 1)
        domain = UnifiedBankDomain(resources, interleave_bytes=16)
        report = domain.simulate([
            self.request("a", "ia", 0), self.request("b", "weight", 0),
            self.request("c", "oa", 0, kind="write"),
        ])
        self.assertEqual(report.per_bank_max_queue_depth[0], 1)

    def test_request_buffer_must_cover_ports(self):
        with self.assertRaisesRegex(ValueError, "cover all Bank ports"):
            ResourceBudget(4, 4096, 64, 2, 1)

    def test_simulation_is_input_order_deterministic(self):
        requests = [self.request("b", "weight", 0), self.request("a", "ia", 0)]
        self.assertEqual(self.domain.simulate(requests), self.domain.simulate(reversed(requests)))

    def test_invalid_preferred_bank_is_rejected(self):
        request = UnifiedMemoryRequest("x", 0, "ia", "x", 0, 16, preferred_banks=(4,))
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.domain.simulate([request])

    def test_report_adapter_rejects_total_accounting_drift(self):
        report = {"ComputeCycles": 100, "BankStallCycles": 10, "TotalCycles": 999}
        with self.assertRaisesRegex(ValueError, "accounting mismatch"):
            policy_result_from_report("static", "static", self.resources, report, (1, 1, 2))

    def test_report_adapter_builds_common_policy_result(self):
        report = {"ComputeCycles": 100, "BankStallCycles": 10, "TotalCycles": 110}
        result = policy_result_from_report(
            "runtime", "runtime_dynamic", self.resources, report, (1, 1, 2)
        )
        self.assertEqual(result.total_cycles, 110)


if __name__ == "__main__":
    unittest.main()
