# Initialize extractor
extractor = FOLIOTypeExtractor()

# Example email text
email_text = """
Dear John,

I am writing to inform you about the upcoming court hearing scheduled for next month.
The plaintiff, Mr. Smith, has submitted a motion requesting additional time to prepare
the necessary documents. The defendant's legal team has agreed to this extension.

Please find attached the updated court filing and the judge's order.

Best regards,
Legal Team
"""

# Extract and validate types
results = extractor.validate_email_content(email_text)

# Print results
print("Validation Results:")
print(f"Is Valid: {results['is_valid']}")
print("\nFound Types:")
for branch, types in results["found_types"].items():
    print(f"{branch}: {', '.join(types)}")
print("\nMissing Types:")
for branch, types in results["missing_types"].items():
    print(f"{branch}: {', '.join(types)}")
print("\nSuggestions:")
for suggestion in results["suggestions"]:
    print(f"- {suggestion}")
