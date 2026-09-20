/// Pure Dart CTC Greedy Decoder for Sequence Recognition (PP-OCR / CRNN / SVTR).
class CtcDecoder {
  final List<String> characterDict;
  final int blankIndex;

  CtcDecoder({
    required this.characterDict,
    this.blankIndex = 0,
  });

  /// Decodes sequence probabilities of shape [timeSteps, numClasses].
  /// Returns a tuple containing the decoded text and the average confidence score.
  CtcResult decode(List<List<double>> probabilities) {
    if (probabilities.isEmpty) {
      return const CtcResult(text: '', confidence: 0.0);
    }

    final indices = <int>[];
    final confidences = <double>[];

    int prevIndex = -1;
    for (final step in probabilities) {
      // Find argmax
      int maxIdx = 0;
      double maxProb = step[0];
      for (int i = 1; i < step.length; i++) {
        if (step[i] > maxProb) {
          maxProb = step[i];
          maxIdx = i;
        }
      }

      // CTC Collapse: collapse repeated characters and ignore blanks
      if (maxIdx != blankIndex && maxIdx != prevIndex) {
        indices.add(maxIdx);
        confidences.add(maxProb);
      }
      prevIndex = maxIdx;
    }

    if (indices.isEmpty) {
      return const CtcResult(text: '', confidence: 0.0);
    }

    // Convert indices to characters using dictionary
    final sb = StringBuffer();
    for (final idx in indices) {
      // Note: in PaddleOCR, character_dict index 0 is mapped to dict[0], with blank usually at 0 or numClasses - 1.
      // Adjusting for 1-based index if blank is at 0:
      final charIdx = idx - 1;
      if (charIdx >= 0 && charIdx < characterDict.length) {
        sb.write(characterDict[charIdx]);
      }
    }

    final avgConf = confidences.isNotEmpty
        ? confidences.reduce((a, b) => a + b) / confidences.length
        : 0.0;

    return CtcResult(text: sb.toString(), confidence: avgConf);
  }
}

class CtcResult {
  final String text;
  final double confidence;

  const CtcResult({required this.text, required this.confidence});
}
