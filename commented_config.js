{
  "seed": 12345, // Random seed for the generator.
  "try_disjoint_splits": false, // If true, it will try to partition positive and negative sets into disjoint train, validation and test set splits.
  "minimum_split_samples": 32, // Minimum number of samples guaranteed for each split. If the cardianality of the sets for a given task is below this value, it behaves as if try_disjoint_splits were false.
  "canvas_size": [128, 128], // Image size (w, h)
  "padding": 10, // Padding in pixels around the generated shapes (Only applied to outermost level).
  "bg_color": "#7f7f7f", // Background color.
  "colors": { // Dictionary of colors. #rrggbb
     "red": "#ff0000",
     "green": "#00ff00",
     "blue": "#0000ff",
     "magenta": "#ff00ff",
     "cyan": "#00ffff",
     "yellow": "#ffff00"
  },
  "sizes": { // Dictionary of sizes.
      "small": 10,
      "large": 25
  },
  "size_noise": 5, // Uniform noise for sizes. Make sure each size +- this value is still unambiguous (eg. [10, 20, 30] should have noise < 5, because 10+5 and 20-5 both collapse to 15).
  "h_sigma": 0.01, // Standard deviation for Gaussian noise in color hues. Make sure sigma is small enough to guarantee color +- 3sigma is still perceptually the same class (eg. "yellow" according to a human observer)
  "s_sigma": 0.2, // Standard deviation for Gaussian noise in color saturation.
  "v_sigma": 0.2 // // Standard deviation for Gaussian noise in color value.
}