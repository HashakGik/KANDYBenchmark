// A curriculum is a list of tasks, each have a name, some parameters and two lists, of positive and negative samples.
[
  {
    "task_name": "triangles or small green circles", // Task name.
    "samples": 1000, // How many samples this task is composed of.
    "alpha": 0.0, // Probability of sampling from past tasks (geometric distribution)
    "beta": 0.1, // Minimum probability of supervision.
    "gamma": 0.5, // Initial probability of supervision. This value will be exponentially decayed up to beta.
    "positive_samples": [ // List of positive samples. Since this list is randomly sampled, these implicitly define a disjunction
                          // (ie. the list ["triangle", "square"] will produce either a triangle or a square, never both in the same image.
      {
        "shape": "triangle", // Shape: any, triangle, circle, square
        "color": "any", // Color: any, red, green, blue, cyan, magenta, yellow
        "size": "any" // Size: any, small, large

        // Shape e color can also be defined by using:
        // not_X -> anything, except X (eg. not_red -> [green, blue, cyan, magenta, yellow]
        // A|B -> A or B
        // These two mini-rules are not composed (it is not possible to write not_A|B).
      },
      {
      "shape": "circle",
      "color": "green",
      "size": "small"
      }
      /* Other than base objects, there can be composite objects, which recursively compose lists of objects.
      {"in": [...]} -> Objects are centered and drawn at the same position (note that a large object can cover entirely a small one, if drawn after it).
      {"random": [...]} -> Objects are placed randomly, without checking for occlusions.
      {"stack": [...]} -> Objects are stacked top to bottom and horizontally-centered.
      {"side_by_side": [...]} -> Objects are placed horizontally, left to right, and vertically-centered.
      {"grid": [...]} -> Objects are placed in an nxn grid (where n = ceil(sqrt(len(list)))), left to right, top to bottom.
      {"diagonal_ul_lr": [...]} -> Objects are placed diagonally, from the upper left corner to the lower right.
      {"diagonal_ll_ur": [...]} -> Objects are placed diagonally, from the lower left corner to the upper right.

      With the exception of "in" and "random", each construct preallocates the available space for its children
      (eg. a nxn grid with wxh available space will provide each of its children w/n x h/n space).
      This behavior is guaranteed only until the canvas space is enough to draw the largest atomic shape (sizeof("large") + maximum size_noise),
      therefore if the hierarchy grows too large, compared to the original canvas size, objects may overlap.
      For "in" and "random" overlap is expected under any circumstance.

      Composite constructs also take an optional property "shuffled": true/false determining whether the inner list should be drawn in order or not.
      eg.
      {
        "shuffled": true,
        "stack": [
            { "shape": "circle", "color": "red", "size": "any"},
            { "shape": "circle", "color": "green", "size": "any"},
            { "shape": "circle", "color": "blue", "size": "any"},
        ]
      }
      This rule will produce three circles vertically stacked, but their color will be randomly permuted.
      If, on the other hand, shuffled was false (or undefined), the order would always be red on top of green on top of blue.
       */
    ],
    "negative_samples": [ // Same rules as positive list. There is no shortcut "not_positive" because it would be ambiguous in case of composite constructs
      {
        "shape": "not_triangle",
        "color": "any",
        "size": "any"
      }
    ],
    "noisy_color": true, // La hue del colore va alterata casualmente?
    "noisy_size": false // La dimensione deve essere costante o va alterata casualmente?
  },
  {
    "task_name": "complex task",
    "samples": 100,
    "alpha": 0.5,
    "beta": 0.1,
    "gamma": 0.2,
    "positive_samples": [
      {
        "shuffled": true,
        "grid": [
            {
                "diagonal_ul_lr": [
                {
                "shape": "any",
                "color": "any",
                "size": "any"
            },
            {
                "shape": "any",
                "color": "any",
                "size": "any"
            }
            {
                "shape": "any",
                "color": "any",
                "size": "any"
            }
                ]
            },
            {
                "stack": [
                    {
                "shape": "square",
                "color": "any",
                "size": "any"
            },
            {
                "shape": "square",
                "color": "any",
                "size": "any"
            }
                ]
            },
            {
                "shape": "circle",
                "color": "red",
                "size": "any"
            },
            {
                "side_by_side":
                [
                    {
                "shape": "circle",
                "color": "blue",
                "size": "small"
            },
            {
                "stack":
                [
                    {
                "shape": "circle",
                "color": "yellow",
                "size": "small"
            },
            {
                "shape": "circle",
                "color": "green",
                "size": "small"
            },
            {
                "shape": "circle",
                "color": "yellow",
                "size": "small"
            }
                ]
            },
            {
                "shape": "circle",
                "color": "blue",
                "size": "small"
            }
                ]
            }
        ]
      },
      {
            "shape": "circle",
            "color": "red",
            "size": "any"
      }
    ],
    "negative_samples": [
      {
        "shape": "triangle",
        "color": "not_red",
        "size": "any"
      }
    ],
    "noisy_color": false,
    "noisy_size": false
  }
]