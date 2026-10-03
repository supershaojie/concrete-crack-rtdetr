"""An interrupted pilot must report its saved epochs without claiming completion."""
from pathlib import Path
import tempfile
import unittest
from support import write_json
from run import summary


class ReportingTests(unittest.TestCase):
    def test_interrupted_run_retains_completed_epoch_count(self):
        with tempfile.TemporaryDirectory() as td:
            run=Path(td);(run/'native').mkdir()
            write_json(run/'native/epoch_state.json',{'completed_epochs':7,'best_epoch':5,'end_reason':'running'})
            write_json(run/'status.json',{'stages':{'train':{'started':'2026-10-03T01:00:00Z','status':'interrupted','process_exit_code':130}}})
            summary(run)
            import json
            result=json.loads((run/'summary.json').read_text())
            self.assertEqual(result['training']['completed_epochs'],7)
            self.assertEqual(result['training']['completion'],'incomplete')
            self.assertEqual(result['training']['latest_train_attempt']['status'],'interrupted')
            self.assertEqual(result['val'],'NOT_EXECUTED')
            self.assertEqual(result['test'],'NOT_EXECUTED')
            self.assertEqual(result['selected_best'],'NOT_SELECTED')


if __name__=='__main__':
    unittest.main(verbosity=2)
