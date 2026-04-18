from evaluation.metrics import caculate_accuracy


def eval(predicted_sequences, ground_truths):
    accuracy = caculate_accuracy(predicted_sequences, ground_truths)
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result
import re

def extract_first_number(text):
    """Extract the first number or numerical answer from text"""
    if not text:
        return None
        
    # Try to extract independent numbers
    match = re.search(r'\b(\d+)\b', text)
    if match:
        return match.group(1)
    
    # Try to match 'answer: 123' format
    match = re.search(r'[Aa]nswer[:\s]+(\d+)', text)
    if match:
        return match.group(1)
        
    # If it is a text answer (like 'June'), keep it as is
    return text.strip()

def eval(predicted_sequences, ground_truths):
    """Evaluate NumGLUE-ds results, directly compare numerical or text answers"""
    correct = 0
    valid_count = 0
    
    for i in range(len(predicted_sequences)):
        prediction = predicted_sequences[i]
        target = ground_truths[i]
        
        if not prediction or not target:
            continue
            
        # Extract the first number or answer from prediction
        pred_value = extract_first_number(prediction)
        target_value = target.strip()
        
        valid_count += 1
        if pred_value == target_value:
            correct += 1
    
    if valid_count == 0:
        accuracy = 0
    else:
        accuracy = correct / valid_count
    
    print(f"NumGLUE-ds accuracy: {accuracy:.4f} (correct: {correct}/{valid_count})")
    evaluation_result = {"accuracy": accuracy}
    return evaluation_result