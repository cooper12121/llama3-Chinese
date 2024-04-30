import evaluate
print(evaluate.load('pin').compute(references=['hello'], predictions=['hello']))