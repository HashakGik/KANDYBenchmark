import logging
from curriculum_generator import CurriculumGenerator
import os
import shutil

from PIL import Image
import numpy as np

# Simple demo for the curriculum generator class. It saves samples in two folders:
# - samples/curriculum: each sample is generated in teacher mode and numerated progressively in the same order the teacher would provide them.
#       There is no distinction between current and previous tasks, and supervised and unsupervised images are given together (_u = unsupervised, _a/p/n = supervised triplet).
# - samples/sets/{train,val,test}/n: supervised batches for each task 0..n (_a/p/n = supervised triplet).

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    with open("config.json", "r") as file:
        config = file.read()


    with open("curriculum.json", "r") as file:
        curriculum = file.read()

    cg = CurriculumGenerator(config, curriculum, logger)

    shutil.rmtree("samples", ignore_errors=True)
    os.makedirs("samples/curriculum", exist_ok=True)
    os.makedirs("samples/sets", exist_ok=True)
    os.makedirs("samples/sets/train", exist_ok=True)
    os.makedirs("samples/sets/val", exist_ok=True)
    os.makedirs("samples/sets/test", exist_ok=True)

    os.makedirs("samples/curriculum/train", exist_ok=True)
    os.makedirs("samples/curriculum/val", exist_ok=True)
    os.makedirs("samples/curriculum/test", exist_ok=True)

    for split in ["train", "val", "test"]:
        for i in range(len(cg.tasks)):
            os.mkdir("samples/sets/{}/{}".format(split, i))
            for j, sample in enumerate(cg.get_batch(i, 32, split)): # NOTE: get_batch() can be called an infinite number of times, since it randomly samples the split sets.
                (s_a, n_a), (s_p, n_p), (s_n, n_n) = sample
                s_a.save("samples/sets/{}/{}/{:04d}_a.png".format(split, i, j), "PNG")
                s_p.save("samples/sets/{}/{}/{:04d}_p.png".format(split, i, j), "PNG")
                s_n.save("samples/sets/{}/{}/{:04d}_n.png".format(split, i, j), "PNG")

    for split in ["train", "val", "test"]:
        for i, x in enumerate(cg.generate_curriculum(split, batch_size=1)): # NOTE: generate_curriculum() acts as a finite stream, but samples are still chosen randomly.
            tid, a, p, n = x
            if np.sum(p) > 0 and np.sum(n) > 0:
                Image.fromarray(a[0]).save("samples/curriculum/{}/{:02d}_{:04d}_a.png".format(split, tid[0], i), "PNG")
                Image.fromarray(p[0]).save("samples/curriculum/{}/{:02d}_{:04d}_p.png".format(split, tid[0], i), "PNG")
                Image.fromarray(n[0]).save("samples/curriculum/{}/{:02d}_{:04d}_n.png".format(split, tid[0], i), "PNG")
            else:
                Image.fromarray(a[0]).save("samples/curriculum/{}/{:02d}_{:04d}_u.png".format(split, tid[0], i), "PNG")