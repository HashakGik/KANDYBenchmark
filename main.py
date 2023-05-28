import logging
from curriculum_generator import CurriculumGenerator
import os
import shutil

# Simple demo for the curriculum generator class. It saves samples in two folders:
# - samples/curriculum: each sample is generated in teacher mode and numerated progressively in the same order the teacher would provide them.
#       There is no distinction between current and previous tasks, and supervised and unsupervised images are given together (_u = unsupervised, _a/p/n = supervised triplet).
# - samples/test_sets/n: a supervised batch in test set mode (_a/p/n = supervised triplet).

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    with open("curriculum.json", "r") as file:
        config = file.read()

    cg = CurriculumGenerator(config, (128, 128), 10, logger)

    shutil.rmtree("samples")
    os.makedirs("samples/curriculum", exist_ok=True)
    os.makedirs("samples/test_sets", exist_ok=True)

    for i in range(len(cg.tasks)):
        os.mkdir("samples/test_sets/{}".format(i))
        for j, sample in enumerate(cg.get_test_set(i, 32)):
            (s_a, n_a), (s_p, n_p), (s_n, n_n) = sample
            s_a.save("samples/test_sets/{}/{:04d}_a.png".format(i, j), "PNG")
            s_p.save("samples/test_sets/{}/{:04d}_p.png".format(i, j), "PNG")
            s_n.save("samples/test_sets/{}/{:04d}_n.png".format(i, j), "PNG")


    for i, x in enumerate(cg.generate_curriculum()):
        a, p, n = x
        if p is not None and n is not None:
            a.save("samples/curriculum/{:04d}_a.png".format(i), "PNG")
            p.save("samples/curriculum/{:04d}_p.png".format(i), "PNG")
            n.save("samples/curriculum/{:04d}_n.png".format(i), "PNG")
        else:
            a.save("samples/curriculum/{:04d}_u.png".format(i), "PNG")