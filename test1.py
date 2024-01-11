import json
import os

def create_html_with_json(json_data, input_template_path, output_folder, output_filename):
    # Read the HTML template
    with open(input_template_path, 'r') as file:
        html_template = file.read()

    # Convert JSON data to a JavaScript variable format
    json_string = json.dumps(json_data, indent=4)
    js_var_declaration = f"var jsonData = {json_string};"

    # Replace placeholder with JSON data
    #html_with_json = html_template.replace("// Sample JSON data", js_var_declaration)
    html_with_json = html_template.replace('%%JSON_PLACEHOLDER%%', js_var_declaration)

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Write the modified HTML to a new file
    print(html_with_json)

    # output_file_path = os.path.join(output_folder, output_filename)
    # with open(output_file_path, 'w') as output_file:
    #     output_file.write(html_with_json)

    # return output_file_path


# Example usage
json_array = [
 {
  "level1": {
    "subLevel1": {
      "item1": {
        "attribute1": "value1",
        "attribute2": "value2",
        "attribute3": "value3"
      },
      "item2": {
        "attribute1": "value4",
        "attribute2": "value5",
        "attribute3": "value6"
      }
    },
    "subLevel2": {
      "item3": {
        "attribute1": "value7",
        "attribute2": "value8",
        "attribute3": "value9"
      },
      "item4": {
        "attribute1": "value10",
        "attribute2": "value11",
        "attribute3": "value12"
      }
    }
  },
  "level2": {
    "subLevel3": {
      "item5": {
        "attribute1": "value13",
        "attribute2": "value14",
        "attribute3": "value15"
      },
      "item6": {
        "attribute1": "value16",
        "attribute2": "value17",
        "attribute3": "value18"
      }
    },
    "subLevel4": {
      "item7": {
        "attribute1": "value19",
        "attribute2": "value20",
        "attribute3": "value21"
      },
      "item8": {
        "attribute1": "value22",
        "attribute2": "value23",
        "attribute3": "value24"
      }
    }
  }
}
    ]

OUTPUT_FOLDER = '../utilities/' 
input_template_path = '../utilities/label_explorer_template.html'
output_folder = OUTPUT_FOLDER
output_filename = 'output.html'

# Create and save the HTML file
output_html_path = create_html_with_json(json_array, input_template_path, output_folder, output_filename)
print(f"HTML file created at: {output_html_path}")