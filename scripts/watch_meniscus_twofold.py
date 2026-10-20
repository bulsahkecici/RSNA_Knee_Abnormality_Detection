"""Watch remote training, download numeric evidence, and perform the paired review."""
import argparse,fcntl,json,time
from pathlib import Path
import review_meniscus_twofold as reviewer
from check_public_r6 import cli
from review_meniscus_twofold import FOLDER,review
KERNEL='bulsahkecici/rsna-meniscus-twofold-20261020'
def main():
    global FOLDER,KERNEL
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-folder',type=Path)
    args=parser.parse_args()
    if args.run_folder:
        FOLDER=args.run_folder.resolve()
        assert FOLDER.is_relative_to(reviewer.ROOT/'artifacts/kaggle')
        metadata=json.loads((FOLDER/'kernel-metadata.json').read_text())
        KERNEL=metadata['id']
        assert KERNEL.startswith('bulsahkecici/')
        reviewer.FOLDER=FOLDER
    lock=(FOLDER/'watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.monotonic()+8*3600
    while time.monotonic()<deadline:
        try:status=cli('kernels','status',KERNEL+'/1')
        except Exception as exc:
            (FOLDER/'last-query-error.txt').write_text(repr(exc));time.sleep(45);continue
        (FOLDER/'status.txt').write_text(status);print(status.strip(),flush=True)
        if 'COMPLETE' in status:
            cli('kernels','output',KERNEL+'/1','-p',str(FOLDER/'output'),'--force','--quiet',
                '--file-pattern',r'(^|/)(experiment.json|meniscus-metrics.json|meniscus-predictions.jsonl|handshake.json|.*\.log)$')
            result=review()
            state={'status':'REVIEWED','kernel':KERNEL,'version':1,'decision':result['decision'],
                   'submit_authorized':True,'submission_made':False,
                   'next':'reference complementarity and offline packaging' if result['decision']=='NEEDS_REFERENCE_COMPLEMENTARITY' else 'retain 0.958 reference; do not submit failed hypothesis'}
            (FOLDER/'campaign-state.json').write_text(json.dumps(state,indent=2)+'\n')
            return
        if any(s in status for s in ('ERROR','CANCEL','FAILED')):
            cli('kernels','output',KERNEL+'/1','-p',str(FOLDER/'output'),'--force','--quiet','--file-pattern',r'.*\.log$')
            raise RuntimeError(status)
        time.sleep(45)
    raise TimeoutError('remote experiment did not finish')
if __name__=='__main__':main()
