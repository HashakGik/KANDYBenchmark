import json
import numpy as np
import colorsys
import re
import math
import itertools

from PIL import Image, ImageDraw

# Class encoding a single task. It takes the global configuration, task parameters, a list of past tasks, a seeded random number generator, and optionally a logger to print debug informations.
class Task:
    def __init__(self, config, task_specs, task_id, past_tasks, rng, logger=None):
        self.name = task_specs["task_name"] # Task name.
        self.alpha = task_specs["alpha"] # Sampling probability for previous tasks.
        self.beta = task_specs["beta"] # Minimum supervision probability.
        self.gamma = task_specs["gamma"] # Maximum supervision probability.
        self.total_samples = task_specs["samples"] # Number of samples to generate for this task.
        self.noisy_color = task_specs["noisy_color"] # Should the color be altered with noise?
        self.noisy_size = task_specs["noisy_size"] # Should the size be altered with noise?
        
        self.task_id = task_id

        # Color noise parameters: the RGB color is converted to HSV and then 0-mean gaussian noise is injected into each coordinate. These parameters are the standard deviation of each gaussian.
        # Values should be chosen by trial and error to preserve perceptual semantics for the entire palette of colors (ie. a human can classify a noisy yellow still as "yellow").
        self.h_sigma = config["h_sigma"]
        self.s_sigma = config["s_sigma"]
        self.v_sigma = config["v_sigma"]

        self.bg_color = config["bg_color"] # Background color for every image.

        # Size noise parameter. A uniform random value from -noise to +noise is added to size. (eg. a base size of 25 pixels with a noise of 5 can range from 20 to 30 pixels).
        self.size_noise = config["size_noise"]

        # Available base shapes. Extending this dictionary requires to rewrite sample_base_object() as well.
        self.shapes = ["triangle", "square", "circle"]

        # Available colors. This dictionary can be extended arbitrarily, as long as noise is adjusted accordingly.
        self.colors = config["colors"]

        # Available sizes. This dictionary can be extended arbitrarily, as long as noise is adjusted accordingly.
        self.sizes = config["sizes"]

        # Available compositions. Extending this list requires to write new composition methods and to modify sample().
        self.compositions = ["in", "stack", "side_by_side", "diagonal_ul_lr", "diagonal_ll_ur", "grid", "random"]

        self.past_tasks = past_tasks

        self.canvas_size = config["canvas_size"]
        self.padding = config["padding"]
        self.minimum_split_samples = config["minimum_split_samples"]

        self.rng = rng
        self.logger = logger

        self.positive_samples, self.card_p = self.parse_rules(task_specs["positive_samples"])
        self.negative_samples, self.card_n = self.parse_rules(task_specs["negative_samples"])

        self.train_split = task_specs["train_split"]
        self.val_split = task_specs["val_split"]

        self.random_sampling = not config["try_disjoint_splits"] # Whether it should sample randomly from the entire definition sets.
        failed_tests = {}

        if not self.random_sampling:
            # Expand rule sets into every possible combination (ignoring random shuffles for tractability). For deep compositions (> 2) or large sets it may require a huge amount of memory.
            self.p_set = self.expand_rules(self.positive_samples)
            self.n_set = self.expand_rules(self.negative_samples)

            train_size_p = int(len(self.p_set) * self.train_split)
            val_size_p = int(len(self.p_set) * self.val_split)
            test_size_p = len(self.p_set) - train_size_p - val_size_p
            train_size_n = int(len(self.n_set) * self.train_split)
            val_size_n = int(len(self.n_set) * self.val_split)
            test_size_n = len(self.n_set) - train_size_n - val_size_n

            # If any of the positive/negative train/val/test sets has fewer samples than config["minimum_split_samples"], it defaults to random sampling.
            tests = {"Positive training": train_size_p, "Positive validation": val_size_p, "Positive test": test_size_p, "Negative training": train_size_n, "Negative validation": val_size_n, "Negative test": test_size_n}
            for k, v in tests.items():
                if v < self.minimum_split_samples:
                    failed_tests[k] = v
                    self.random_sampling = True

        if self.random_sampling:
            if len(failed_tests) > 0:
                self.log("Some sets are too small for task {} ({}). Defaulting to random sampling.".format(self.name, ", ".join(["{} = {}".format(k, v) for k, v in failed_tests.items()])), "warning")
        else:
            # If every test is successful, create disjoint train/val/test sets.
            self.rng.shuffle(self.p_set)
            self.rng.shuffle(self.n_set)

            self.train_p = self.p_set[0:train_size_p]
            self.val_p = self.p_set[train_size_p:train_size_p + val_size_p]
            self.test_p = self.p_set[train_size_p + val_size_p:]

            self.train_n = self.n_set[0:train_size_n]
            self.val_n = self.n_set[train_size_n:train_size_n + val_size_n]
            self.test_n = self.n_set[train_size_n + val_size_n:]



    # Expands a tree of compositional rules into a list of objects.
    def expand_rules(self, rules):
        full_set = []

        for r in rules:
            if len(set(r.keys()).intersection(self.compositions)) == 0:
                # Base case.
                for sh in r["shape"]:
                    for c in r["color"]:
                        for si in r["size"]:
                            full_set.append({"shape": sh, "color": c, "size": si})
            else:
                # Inductive case.
                for k in r.keys():
                    if k != "shuffled":
                        transposed_part_set = [self.expand_rules([sub_rule]) for sub_rule in r[k]]
                        part_set = list(map(list, itertools.product(*transposed_part_set))) # Deep magic. The result of the recursive call is a list of lists, where the outermost is a sequence of objects and the innermost is a set of options.
                                                                                            # We however need a list of lists, where the outermost is a set of options and the innermost is a sequence of objects.

                        for p in part_set:
                            full_set.append({k: p, "shuffled": r["shuffled"]})
        return full_set


    # Simple logging method.
    def log(self, message, level="debug"):
        if self.logger is not None:
            if level == "debug":
                self.logger.debug(message)
            elif level == "warning":
                self.logger.warning(message)
            else:
                self.logger.error(message)

    # JSON validation: base rules. It also expands not_ and | operators.
    def parse_base_rule(self, r):

        tmp = {}

        if r["shape"] == "any":
            tmp["shape"] = self.shapes
        elif r["shape"].startswith("not_"):
            tmp["shape"] = list(set(self.shapes).difference([r["shape"][len("not_"):]]))
        else:
            tmp["shape"] = r["shape"].split("|")
        if r["color"] == "any":
            tmp["color"] = list(self.colors.keys())
        elif r["color"].startswith("not_"):
            tmp["color"] = list(set(self.colors.keys()).difference([r["color"][len("not_"):]]))
        else:
            tmp["color"] = r["color"].split("|")
        if r["size"] == "any":
            tmp["size"] = list(self.sizes.keys())
        elif r["size"].startswith("not_"):
            tmp["size"] = list(set(self.colors.keys()).difference([r["size"][len("not_"):]]))
        else:
            tmp["size"] = r["size"].split("|")

        tmp["shape"] = list(set(tmp["shape"]))
        tmp["color"] = list(set(tmp["color"]))
        tmp["size"] = list(set(tmp["size"]))

        assert len(set(tmp["shape"]).intersection(self.shapes)) > 0 and len(set(tmp["shape"]).union(self.shapes)) == len(
            self.shapes), "Shapes should be a subset of {}, received {}".format(self.shapes, tmp["shape"])
        assert len(set(tmp["color"]).intersection(self.colors.keys())) > 0 and len(set(tmp["color"]).union(self.colors.keys())) == len(
            self.colors.keys()), "Colors should be a subset of {}, received {}".format(self.colors.keys(), tmp["color"])
        assert len(set(tmp["size"]).intersection(self.sizes.keys())) > 0 and len(set(tmp["size"]).union(self.sizes.keys())) == len(
            self.sizes.keys()), "Sizes should be a subset of {}, received {}".format(self.sizes.keys(), tmp["size"])

        cardinality = len(tmp["shape"]) * len(tmp["color"]) * len(tmp["size"])

        return tmp, cardinality

    # JSON validation: recursive rules.
    def parse_rules(self, rules):
        out = []

        cardinality = 0

        for r in rules:
            assert len(set(r.keys()).intersection(self.compositions)) <= 1, "There can be at most one composition operator at this level, found {}".format(set(r.keys()).intersection(self.compositions))

            if len(set(r.keys()).intersection(self.compositions)) == 0:
                # Base case: the rule under consideration is a base shape.
                b, sub_cardinality = self.parse_base_rule(r)
                out.append(b)
                if cardinality == 0:
                    cardinality = sub_cardinality
                else:
                    cardinality *= sub_cardinality
            else:
                # Inductive case: the rule under consideration is a composition.
                c = list(set(r.keys()).intersection(self.compositions))[0]
                rules, sub_cardinality = self.parse_rules(r[c])
                tmp = {c : rules}
                if "shuffled" in r.keys():
                    tmp["shuffled"] = r["shuffled"]
                else:
                    tmp["shuffled"] = False

                cardinality += sub_cardinality * (1 if not tmp["shuffled"] else len(rules))


                out.append(tmp)

        return out, cardinality

    # Compute random (uniform) size noise.
    def inject_size_noise(self, size):
        if self.size_noise > 0:
            rnd = self.rng.uniform(-self.size_noise, self.size_noise)
        else:
            rnd = 0
        return max(1, size + rnd)

    # colorsys requires lists, PIL requires #rrggbb strings. These utilities handle conversions.

    def _rgb_to_list(self, string):
        assert re.match("#[0-9a-f]{6}", string, flags=re.IGNORECASE) is not None
        rgb = re.search("#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})", string, flags=re.IGNORECASE).groups()
        rgb = [int(x, 16) for x in rgb]
        return rgb

    def _list_to_rgb(self, rgb):
        assert len(rgb) == 3
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    # Inject gaussian noise into HSV coordinates.
    def inject_color_noise(self, rgb):
        hsv = colorsys.rgb_to_hsv(*self._rgb_to_list(rgb))
        h = hsv[0]  # h in [0.0, 1.0]
        s = hsv[1]  # s in [0.0, 1.0]
        v = hsv[2]  # v in [0, 255]
        if self.h_sigma > 0.0:
            rnd = self.rng.normal(0, self.h_sigma)
            h = max(0, min(1, h + rnd * 1.0))
        if self.s_sigma > 0.0:
            rnd = self.rng.normal(0, self.s_sigma)
            s = max(0, min(1, s + rnd * 1.0))
        if self.v_sigma > 0.0:
            rnd = self.rng.normal(0, self.v_sigma)
            v = max(0, min(255, int(v + rnd * 255)))

        return self._list_to_rgb(colorsys.hsv_to_rgb(h, s, v))

    # Randomly sample a base object from the sample list and draw it on a canvas_size image.
    def sample_base_object(self, sample, canvas_size):
        if isinstance(sample["shape"], list):
            shape = self.rng.choice(sample["shape"])
        else:
            shape = sample["shape"]
        if isinstance(sample["color"], list):
            color_name = self.rng.choice(sample["color"])
        else:
            color_name = sample["color"]
        if isinstance(sample["size"], list):
            size_name = self.rng.choice(sample["size"])
        else:
            size_name = sample["size"]

        if self.noisy_color:
            color = self.inject_color_noise(self.colors[color_name])
        else:
            color = self.colors[color_name]
        if self.noisy_size:
            size = self.inject_size_noise(self.sizes[size_name])
        else:
            size = self.sizes[size_name]

        if canvas_size[0] < max(self.sizes.values()) + self.size_noise or canvas_size[1] < max(self.sizes.values()) + self.size_noise:
            self.log("Canvas size too small. Defaulting to ({},{})".format(max(self.sizes.values()) + self.size_noise, max(self.sizes.values()) + self.size_noise), "warning")
            canvas_size = (max(self.sizes.values()) + self.size_noise, max(self.sizes.values()) + self.size_noise)

        logstring = "({}-{}-{})".format(shape, color_name, size_name)
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(bitmap)

        bounding_box = (
        (canvas_size[0] / 2 - size / 2, canvas_size[1] / 2 - size / 2),
        (canvas_size[0] / 2 + size / 2, canvas_size[1] / 2 + size / 2))

        # PIL regular_polygon wants the coordinates of the inscribed circle, making polygons too large for the canvas size. Therefore we manually draw our own.

        if shape == "circle":
            draw.ellipse(bounding_box, fill=color, outline=None)
        elif shape == "triangle":
            xy = ((canvas_size[0] / 2 + size / 2 * np.cos(2 * np.pi / 3 * 0 - np.pi / 2),
                   canvas_size[1] / 2 + size / 2 * np.sin(2 * np.pi / 3 * 0 - np.pi / 2)),
                  (canvas_size[0] / 2 + size / 2 * np.cos(2 * np.pi / 3 * 1 - np.pi / 2),
                   canvas_size[1] / 2 + size / 2 * np.sin(2 * np.pi / 3 * 1 - np.pi / 2)),
                  (canvas_size[0] / 2 + size / 2 * np.cos(2 * np.pi / 3 * 2 - np.pi / 2),
                   canvas_size[1] / 2 + size / 2 * np.sin(2 * np.pi / 3 * 2 - np.pi / 2))
                  )
            draw.polygon(xy, fill=color, outline=None)
        elif shape == "square":
            xy = ((canvas_size[0] / 2 - size / 2, canvas_size[1] / 2 - size / 2), (canvas_size[0] / 2 - size / 2, canvas_size[1] / 2 + size / 2), (canvas_size[0] / 2 + size / 2, canvas_size[1] / 2 + size / 2), (canvas_size[0] / 2 + size / 2, canvas_size[1] / 2 - size / 2))
            draw.polygon(xy, fill=color, outline=None)

        return (bitmap, logstring)

    # "in" composition function. Recursively samples each element from the sample list, then draws them in a canvas_size image.
    def sample_in(self, sample, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        logstrings = []

        if sample["shuffled"]:
            self.rng.shuffle(sample["in"])

        for s in sample["in"]:
            bmp, ls = self.sample(s, canvas_size)
            bitmap.paste(bmp, mask=bmp)
            logstrings.append(ls)

        return bitmap, "in[{}]".format("-".join(logstrings))

    # "stack" composition function. Recursively samples each element from the sample list, then draws them in a canvas_size image.
    def sample_stack(self, sample, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        child_canvas = ( self.size_noise + max(*self.sizes.values(), canvas_size[0]),
                         self.size_noise + max(*self.sizes.values(), canvas_size[1] // len(sample["stack"])))

        logstrings = []
        bitmaps = []

        if sample["shuffled"]:
            self.rng.shuffle(sample["stack"])

        for s in sample["stack"]:
            bmp, ls = self.sample(s, child_canvas)
            logstrings.append(ls)
            bitmaps.append(bmp)

        step = canvas_size[1] / (len(bitmaps) + 1)

        for i in range(len(bitmaps)):
            x0 = int(0)
            y0 = int((i + 1) * step - child_canvas[1] / 2)

            bitmap.paste(bitmaps[i], (x0, y0), mask=bitmaps[i])

        return bitmap, "stack[{}]".format("-".join(logstrings))

    # "side_by_side" composition function. Recursively samples each element from the sample list, then draws them in a canvas_size image.
    def sample_sbs(self, sample, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        child_canvas = ( self.size_noise + max(*self.sizes.values(), canvas_size[0] // len(sample["side_by_side"])),
                         self.size_noise + max(*self.sizes.values(), canvas_size[1]) )

        logstrings = []
        bitmaps = []

        if sample["shuffled"]:
            self.rng.shuffle(sample["side_by_side"])

        for s in sample["side_by_side"]:
            bmp, ls = self.sample(s, child_canvas )
            logstrings.append(ls)
            bitmaps.append(bmp)

        step = canvas_size[0] / (len(bitmaps) + 1)

        for i in range(len(bitmaps)):
            y0 = int(0)
            x0 = int((i + 1) * step - child_canvas[0] / 2)

            bitmap.paste(bitmaps[i], (x0, y0), mask=bitmaps[i])

        return bitmap, "side_by_side[{}]".format("-".join(logstrings))

    # "diagonal_ul_lr" composition function. Recursively samples each element from the sample list, then draws them in a canvas_size image.
    def sample_ullr(self, sample, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        child_canvas = ( self.size_noise + max(*self.sizes.values(), canvas_size[0] // len(sample["diagonal_ul_lr"])),
                         self.size_noise + max(*self.sizes.values(), canvas_size[1] // len(sample["diagonal_ul_lr"])))

        logstrings = []
        bitmaps = []

        if sample["shuffled"]:
            self.rng.shuffle(sample["diagonal_ul_lr"])

        for s in sample["diagonal_ul_lr"]:
            bmp, ls = self.sample(s, child_canvas)
            logstrings.append(ls)
            bitmaps.append(bmp)

        step = 1 / (len(bitmaps) + 1)

        for i in range(len(bitmaps)):
            d = (i + 1) * step
            x0 = int(d * canvas_size[0] - child_canvas[0] / 2)
            y0 = int(d * canvas_size[1] - child_canvas[1] / 2)

            bitmap.paste(bitmaps[i], (x0, y0), mask=bitmaps[i])

        return bitmap, "diag1[{}]".format("-".join(logstrings))

    # "diagonal_ll_ur" composition function. Recursively samples each element from the sample list, then draws them in a canvas_size image.
    def sample_llur(self, sample, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        child_canvas = ( self.size_noise + max(*self.sizes.values(), canvas_size[0] // len(sample["diagonal_ll_ur"])),
                         self.size_noise + max(*self.sizes.values(), canvas_size[1] // len(sample["diagonal_ll_ur"])))

        logstrings = []
        bitmaps = []

        if sample["shuffled"]:
            self.rng.shuffle(sample["diagonal_ll_ur"])

        for s in sample["diagonal_ll_ur"]:
            bmp, ls = self.sample(s, child_canvas)
            logstrings.append(ls)
            bitmaps.append(bmp)

        step = 1 / (len(bitmaps) + 1)

        for i in range(len(bitmaps)):
            d = (i + 1) * step
            x0 = int(canvas_size[0] - d * canvas_size[0] - child_canvas[0] / 2)
            y0 = int(d * canvas_size[1] - child_canvas[1] / 2)

            bitmap.paste(bitmaps[i], (x0, y0), mask=bitmaps[i])

        return bitmap, "diag2[{}]".format("-".join(logstrings))

    # "grid" composition function. Recursively samples each element from the sample list, then draws them in a canvas_size image.
    def sample_grid(self, sample, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        n = math.ceil(math.sqrt(len(sample["grid"])))

        child_canvas = ( self.size_noise + max(*self.sizes.values(), canvas_size[0] // n),
                         self.size_noise + max(*self.sizes.values(), canvas_size[1] // n))

        logstrings = []
        bitmaps = []

        if sample["shuffled"]:
            self.rng.shuffle(sample["grid"])

        for s in sample["grid"]:
            bmp, ls = self.sample(s, child_canvas)
            logstrings.append(ls)
            bitmaps.append(bmp)


        for i in range(len(bitmaps)):
            j = (i % n)
            k = i // n

            x0 = int(j * canvas_size[0] / n)
            y0 = int(k * canvas_size[1] / n)

            bitmap.paste(bitmaps[i], (x0, y0), mask=bitmaps[i])

        return bitmap, "grid[{}]".format("-".join(logstrings))

    # "random" composition function. Recursively samples each element from the sample list, then draws them in a canvas_size image.
    def sample_random(self, sample, canvas_size):
        bitmap = Image.new('RGBA', canvas_size, (0, 0, 0, 0))

        logstrings = []

        for s in sample["random"]:
            bmp, ls = self.sample(s, canvas_size)
            m = max(self.sizes.values()) + self.size_noise
            x0 = self.rng.randint(m // 2, canvas_size[0] - m // 2) - canvas_size[0] // 2
            y0 = self.rng.randint(m // 2, canvas_size[1] - m // 2) - canvas_size[1] // 2
            x1 = x0 + canvas_size[0]
            y1 = y0 + canvas_size[1]

            bitmap.paste(bmp, (x0, y0, x1, y1), mask=bmp)
            logstrings.append(ls)

        return bitmap, "random[{}]".format("-".join(logstrings))

    # Wrapper for the recursive sampling procedure.
    def sample(self, sample_set, canvas_size):
        if isinstance(sample_set, list):
            sample = self.rng.choice(sample_set)
        else:
            sample = sample_set


        if "in" in sample.keys():
            return self.sample_in(sample, canvas_size)
        elif "stack" in sample.keys():
            return self.sample_stack(sample, canvas_size)
        elif "side_by_side" in sample.keys():
            return self.sample_sbs(sample, canvas_size)
        elif "diagonal_ul_lr" in sample.keys():
            return self.sample_ullr(sample, canvas_size)
        elif "diagonal_ll_ur" in sample.keys():
            return self.sample_llur(sample, canvas_size)
        elif "grid" in sample.keys():
            return self.sample_grid(sample, canvas_size)
        elif "random" in sample.keys():
            return self.sample_random(sample, canvas_size)
        else:
            return self.sample_base_object(sample, canvas_size)

    # Upper level wrapper. Draws the sample and then removes transparency and adds a padding.
    # It returns both the image and a logstring describing the objects in the image (useful for logging or symbolic processing).
    def draw_sample(self, sample_set):
        w = self.canvas_size[0] - self.padding
        h = self.canvas_size[1] - self.padding
        bitmap, logstring = self.sample(sample_set, (w,h))

        out = Image.new('RGBA', self.canvas_size, self.bg_color)
        out.paste(bitmap, (self.padding // 2, self.padding // 2), mask=bitmap)

        return out.convert('RGB'), logstring

    # Outputs a supervised triple (anchor, positive, negative).
    def sample_supervised(self, split=None):
        if self.random_sampling or split is None:
            return self.draw_sample(self.positive_samples), self.draw_sample(self.positive_samples), self.draw_sample(self.negative_samples)
        else:
            assert split in ["train", "val", "test"]
            if split == "train":
                return self.draw_sample(self.train_p), self.draw_sample(self.train_p), self.draw_sample(self.train_n)
            elif split == "val":
                return self.draw_sample(self.val_p), self.draw_sample(self.val_p), self.draw_sample(self.val_n)
            elif split == "test":
                return self.draw_sample(self.test_p), self.draw_sample(self.test_p), self.draw_sample(self.test_n)

    # Outputs an unsupervised sample (anchor, None, None).
    def sample_unsupervised(self, positive, split=None):
        if self.random_sampling or split is None:
            if positive:
                return self.draw_sample(self.positive_samples), None, None
            else:
                return self.draw_sample(self.negative_samples), None, None
        else:
            assert split in ["train", "val", "test"]
            if split == "train":
                if positive:
                    return self.draw_sample(self.train_p), None, None
                else:
                    return self.draw_sample(self.train_n), None, None
            elif split == "val":
                if positive:
                    return self.draw_sample(self.val_p), None, None
                else:
                    return self.draw_sample(self.val_n), None, None
            elif split == "test":
                if positive:
                    return self.draw_sample(self.test_p), None, None
                else:
                    return self.draw_sample(self.test_n), None, None

    # Produces size supervised triples (anchor, positive, negative) as a list of tuples. If converted to numpy the shape would be (size, apn=3, width, height, rgb=3).
    def get_batch(self, size, split=None):
        return [self.sample_supervised(split) for _ in range(size)]

    # Teacher generator. It samples a decision and generates images accordingly. Sampling alternates old tasks and the current one, to avoid starvation.
    # If split=None, the number of samples generated is guaranted to be at least self.total (in case it never samples from past tasks) and at most 2*self.total (in case at each step it decides to sample from past tasks).
    # Otherwise it guarantees self.total * split percentage (as specified for each task in the curriculum JSON).
    def get_decision(self, split="train"):
        self.log("BEGINNING TASK {} (Split: {})".format(self.name, split))

        if self.random_sampling:
            split_samples = self.total_samples
        elif split == "train":
            split_samples = int(self.total_samples * self.train_split)
        elif split == "val":
            split_samples = int(self.total_samples * self.val_split)
        elif split == "test":
            split_samples = self.total_samples - int(self.total_samples * self.train_split) - int(self.total_samples * self.val_split)
        else:
            split_samples = self.total_samples
        self.log("Number of samples: {} (total), {} (current split), alpha: {}, beta: {}, gamma: {}".format(self.total_samples, split_samples, self.alpha, self.beta, self.gamma))
        i = 0

        scale = (np.log(self.gamma / self.beta)) / (self.gamma * split_samples) # Guarantee a minimum probability of beta and a maximum of gamma. See appendix for an explanation.


        while i < split_samples:
            # 1. Extract from a geometric distribution alpha * (1 - alpha)**j the probability of sampling from old task -j.
            # It stops at the first successful decision. If every decision fails, it extracts no samples from the past.
            for j in range(len(self.past_tasks)):
                if self.rng.random() < self.alpha * (1 - self.alpha) ** j:
                    (s_a, logstring_a),(s_p, logstring_p),(s_n, logstring_n) = self.past_tasks[len(self.past_tasks) - j - 1].sample_supervised(split)
                    self.log("SUPERVISED SAMPLE FROM OLD TASK {}: {}".format(self.past_tasks[len(self.past_tasks) - j - 1].name, (logstring_a, logstring_p, logstring_n)))
                    yield self.past_tasks[len(self.past_tasks) - j - 1].task_id, s_a, s_p, s_n
                    break # Stop at the first success.

            # 2. Extract from an exponential distribution the decision of providing a supervised or unsupervised sample.
            # Each sample is guaranteed to be supervised with at least a probability of beta.
            t = i * scale
            if self.rng.random() < self.gamma * np.exp(-self.gamma * t):
                (s_a, logstring_a),(s_p, logstring_p),(s_n, logstring_n) = self.sample_supervised(split)
                self.log("SUPERVISED SAMPLE: {}".format((logstring_a, logstring_p, logstring_n)))
                yield self.task_id, s_a, s_p, s_n
            else:
                if self.rng.randint(0, 2) == 1:
                    (s_a, logstring_a),_, _ = self.sample_unsupervised(True, split)
                    self.log("UNSUPERVISED SAMPLE (POSITIVE): {}".format((logstring_a)))
                else:
                    (s_a, logstring_a), _, _ = self.sample_unsupervised(False, split)
                    self.log("UNSUPERVISED SAMPLE (NEGATIVE): {}".format((logstring_a)))
                yield self.task_id, s_a, None, None

            i += 1

        self.log("END OF TASK {}".format(self.name))


# Curriculum generator class. It wraps multiple tasks into a single object.
class CurriculumGenerator:
    def __init__(self, config, curriculum, logger=None):
        self.tasks = []
        self.current_task = 0

        self.logger = logger
        self.config = {}
        self.parse_config(json.loads(config))
        self.rng = np.random.RandomState(self.config["seed"])
        self.parse_curriculum(json.loads(curriculum))
        


    # Parse the global configuration JSON.
    def parse_config(self, config):
        self.config["seed"] = int(config["seed"])
        self.config["minimum_split_samples"] = max(1, int(config["minimum_split_samples"]))
        self.config["canvas_size"] = (max(32, int(config["canvas_size"][0])), max(32, int(config["canvas_size"][1])))
        self.config["padding"] = max(0, int(config["padding"]))
        self.config["bg_color"] = config["bg_color"]
        self.config["colors"] = config["colors"]
        self.config["sizes"] = config["sizes"]
        self.config["size_noise"] = int(config["size_noise"])
        self.config["h_sigma"] = max(0.0, float(config["h_sigma"]))
        self.config["s_sigma"] = max(0.0, float(config["s_sigma"]))
        self.config["v_sigma"] = max(0.0, float(config["v_sigma"]))
        self.config["try_disjoint_splits"] = config["try_disjoint_splits"]

    # Parse the curriculum JSON.
    def parse_curriculum(self, curriculum):
        for i, c in enumerate(curriculum):
            self.tasks.append(Task(self.config, c, i, self.tasks[0:i], self.rng, self.logger))

    # Reset the generator.
    def reset(self):
        self.current_task = 0

    # Generator for the entire curriculum. It visits each task in order and returns a batch. Optionally corrupts the task id.
    def generate_curriculum(self, split="train", task_id_noise=0.0, batch_size=1):
        self.reset()
        i = 0
        tid_np = np.zeros(batch_size, dtype=np.uint16)
        a_np = np.zeros((batch_size, self.config["canvas_size"][0], self.config["canvas_size"][1], 3), dtype=np.uint8)
        p_np = np.zeros((batch_size, self.config["canvas_size"][0], self.config["canvas_size"][1], 3), dtype=np.uint8)
        n_np = np.zeros((batch_size, self.config["canvas_size"][0], self.config["canvas_size"][1], 3), dtype=np.uint8)

        while self.current_task < len(self.tasks):
            for sample in self.tasks[self.current_task].get_decision(split):

                tid, a, p, n = sample

                tid_np[i] = tid
                a_np[i,:,:,:] = a
                if p is not None and n is not None:
                    p_np[i,:,:,:] = p
                    n_np[i,:,:,:] = n

                i += 1

                if i >= batch_size:
                    tid_np = np.where(self.rng.random(size=batch_size) < task_id_noise, self.rng.randint(len(self.tasks)), tid)

                    yield tid_np, a_np, p_np, n_np
                    i = 0
                    tid_np = np.zeros(batch_size, dtype=np.uint16)
                    a_np = np.zeros((batch_size, self.config["canvas_size"][0], self.config["canvas_size"][1], 3),
                                    dtype=np.uint8)
                    p_np = np.zeros((batch_size, self.config["canvas_size"][0], self.config["canvas_size"][1], 3),
                                    dtype=np.uint8)
                    n_np = np.zeros((batch_size, self.config["canvas_size"][0], self.config["canvas_size"][1], 3),
                                    dtype=np.uint8)

            self.current_task += 1

            if i > 0:
                tid_np[:i] = np.where(self.rng.random(size=i) < task_id_noise, self.rng.randint(len(self.tasks)), tid)
                yield tid_np[:i], a_np[:i,:,:,:], p_np[:i,:,:,:], n_np[:i,:,:,:]

    # Generator for the entire curriculum. Works like generate_curriculum, but samples are sampled randomly from every task (which in turn can decide to provide a previous task sample).
    # Since each task will be exhausted at some point, sampling is not completely i.i.d., with batches later in the curriculum being sampled from fewer and fewer tasks.
    def generate_shuffled_curriculum(self, split="train", task_id_noise=0.0, batch_size=1):
        self.reset()
        i = 0
        task_iterators = [t.get_decision(split) for t in self.tasks]

        tid_np = np.zeros(batch_size, dtype=np.uint16)
        a_np = np.zeros((batch_size, self.config["canvas_size"][0], self.config["canvas_size"][1], 3), dtype=np.uint8)
        p_np = np.zeros((batch_size, self.config["canvas_size"][0], self.config["canvas_size"][1], 3), dtype=np.uint8)
        n_np = np.zeros((batch_size, self.config["canvas_size"][0], self.config["canvas_size"][1], 3), dtype=np.uint8)

        while len(task_iterators) > 0:
            task = self.rng.choice(task_iterators)


            try:
                tid, a, p, n = next(task)

                tid_np[i] = tid
                a_np[i, :, :, :] = a
                if p is not None and n is not None:
                    p_np[i, :, :, :] = p
                    n_np[i, :, :, :] = n

                i += 1

                if i >= batch_size:
                    tid_np = np.where(self.rng.random(size=batch_size) < task_id_noise, self.rng.randint(len(self.tasks), size=batch_size), tid_np)

                    yield tid_np, a_np, p_np, n_np
                    i = 0
                    tid_np = np.zeros(batch_size, dtype=np.uint16)
                    a_np = np.zeros((batch_size, self.config["canvas_size"][0], self.config["canvas_size"][1], 3),
                                    dtype=np.uint8)
                    p_np = np.zeros((batch_size, self.config["canvas_size"][0], self.config["canvas_size"][1], 3),
                                    dtype=np.uint8)
                    n_np = np.zeros((batch_size, self.config["canvas_size"][0], self.config["canvas_size"][1], 3),
                                    dtype=np.uint8)
            except StopIteration:
                task_iterators.remove(task)

        if i > 0:
            tid_np[:i] = np.where(self.rng.random(size=i) < task_id_noise, self.rng.randint(len(self.tasks), size=i), tid_np[:i])
            yield tid_np[:i], a_np[:i,:,:,:], p_np[:i,:,:,:], n_np[:i,:,:,:]

    # Returns a test set for task i, composed of size samples.
    def get_batch(self, i, size, split=None):
        if i < len(self.tasks):
            return self.tasks[i].get_batch(size, split)
        else:
            raise IndexError()

    # Returns a test set for the current task, composed of size samples.
    def get_current_batch(self, size, split=None):
        return self.get_batch(self.current_task, size, split)




"""
Computing the scale value for the exponential distribution:

We cannot use directly a geometric distribution in virtue of the fact that the number of samples is large and probabilities quickly drop to zero.
We use an exponential distribution opportunely limited between 0 and a value such that the minimum probability is beta.

We obtain a continuous variable by normalizing task progression in [0,1]: t = i / self.total, then we rescale this value by:
scale = log(gamma / beta) / gamma.

This scale factor comes from solving the equation:
beta = gamma * e **(-gamma * scale)

This can be easily rewritten as:
log(beta) = log(gamma) - gamma * scale * log(e)
log(beta) = log(gamma) - gamma * scale
log(beta) - log(gamma) = -gamma * scale
(log(gamma) - log(beta)) / gamma = scale
log(gamma / beta) / gamma = scale

gamma * e ** (-gamma * scale * i / total) is guaranteed to have a maximum gamma in i=0 and a minimum beta in i=total.
"""