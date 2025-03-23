map(
  {
    answer_idx: .answer_idx,
    output: (
      .output
      | gsub("<think>(.|\\n)*?</think>"; "")  # Remove <think> tags (multiline safe)
      | (match("The answer is\\s+(?:\\*\\*)?([A-Z])(?:\\*\\*)?(?:\\.|\\b)"; "i").captures[0].string // "" ) # Extract answer, case-insensitive, handles variations, handles the cases that not match
    )
  }
)