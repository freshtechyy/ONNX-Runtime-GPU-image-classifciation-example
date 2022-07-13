#include <iostream>
#include <string>
#include <vector>

#include "dirent.h"
#include "image_classifier.h"

/**
 * @brief Get all the image filenames in a specified directory
 * @param img_dir: the input directory
 * @param img_names: the vector storing all the image filenames
 */
void getAllImageFiles(const std::string &img_dir,
                      std::vector<std::string> &img_names) {
  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir(img_dir.c_str())) != NULL) {
    while ((ent = readdir(dir)) != NULL) {
      std::string filename(ent->d_name);
      if (filename == "." || filename == "..") continue;
      img_names.push_back(filename);
    }
    closedir(dir);
  } else {
    // Failed to open directory
    perror("");
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char **argv) {
  // Create image classifier
  ImageClassifier ic("../models/image_classifier.onnx");

  // Load images in the input directory
  std::string img_dir("../images/");
  std::vector<std::string> img_names;
  getAllImageFiles(img_dir, img_names);

  // Classes
  std::vector<std::string> classes = {"plane", "car",  "bird", "cat",
                                      "deer",  "dog",  "frog", "horse",
                                      "ship",  "truck"};

  // Inference using image classifier
  std::cout << "******* Predicition results below *******" << std::endl;
  for (int i = 0; i < int(img_names.size()); ++i) {
    std::string img_path = img_dir + img_names[i];
    std::cout << "Loaded image: " << img_path << std::endl;
    int cls_idx = ic.Inference(img_path);
    std::cout << "Predicted class: " << classes[cls_idx] << std::endl
              << std::endl;
  }

  std::cout << "Successfully performed image classification" << std::endl;

  return 0;
}