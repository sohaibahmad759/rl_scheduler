import os
import glob
import logging
import time
import torch
from PIL import Image
from transformers import RobertaTokenizer, AlbertTokenizer, BertTokenizer, RobertaForSequenceClassification, AlbertForSequenceClassification, BertForSequenceClassification, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, AutoModelForSeq2SeqLM, AutoImageProcessor, ResNetForImageClassification, EfficientFormerForImageClassification, EfficientFormerImageProcessor, AutoModelForImageClassification


import sys
sys.path.append('../../')
from cluster_scripts.clusterpredictor import ClusterPredictor
from core.common import Event, EventType


def load_model_and_inputs(model_variant_name, device):
    if 'yolov5' in model_variant_name:
        model = torch.hub.load('ultralytics/yolov5', model_variant_name).to(device)
        input_batch_1 = [Image.open('images/cat.jpg')]
        input_batch_2 = [Image.open('images/cat.jpg')] * 2
        input_batch_4 = [Image.open('images/cat.jpg')] * 4
        input_batch_8 = [Image.open('images/cat.jpg')] * 8
    elif 'bert' in model_variant_name:
        if 'roberta' in model_variant_name:
            tokenizer = RobertaTokenizer.from_pretrained(model_variant_name)
            model = RobertaForSequenceClassification.from_pretrained(model_variant_name, cache_dir=cache_dir,
                                                                     return_dict=True).to(device)
        elif 'albert' in model_variant_name:
            tokenizer = AlbertTokenizer.from_pretrained(model_variant_name)
            model = AlbertForSequenceClassification.from_pretrained(model_variant_name, cache_dir=cache_dir,
                                                                    return_dict=True).to(device)
        elif 'prajjwal' in model_variant_name:
            tokenizer = AutoTokenizer.from_pretrained(model_variant_name)
            model = BertForSequenceClassification.from_pretrained(model_variant_name,
                                                                  cache_dir=cache_dir,
                                                                  return_dict=True).to(device)
        elif 'bert' in model_variant_name:
            tokenizer = BertTokenizer.from_pretrained(model_variant_name)
            model = BertForSequenceClassification.from_pretrained(model_variant_name, cache_dir=cache_dir,
                                                                  return_dict=True).to(device)
        else:
            raise Exception(f'unidentified bert variant: {model_variant_name}')
        
        original_sequences = ["Hello, my dog is cute"]
        sequences = original_sequences
        input_batch_1 = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True) for sequence in sequences]).to(device)
        sequences = original_sequences * 2
        input_batch_2 = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True) for sequence in sequences]).to(device)
        sequences = original_sequences * 4
        input_batch_4 = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True) for sequence in sequences]).to(device)
        sequences = original_sequences * 8
        input_batch_8 = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True) for sequence in sequences]).to(device)
    elif 'gpt2' in model_variant_name:
        tokenizer = GPT2Tokenizer.from_pretrained(model_variant_name)
        model = GPT2LMHeadModel.from_pretrained(model_variant_name,
                                                cache_dir=cache_dir,
                                                return_dict=True).to(device)

        original_sequences = ["Hello, my dog is cute"]
        sequences = original_sequences
        input_batch_1 = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True) for sequence in sequences]).to(device)
        sequences = original_sequences * 2
        input_batch_2 = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True) for sequence in sequences]).to(device)
        sequences = original_sequences * 4
        input_batch_4 = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True) for sequence in sequences]).to(device)
        sequences = original_sequences * 8
        input_batch_8 = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True) for sequence in sequences]).to(device)
    elif 't5' in model_variant_name:
        tokenizer = AutoTokenizer.from_pretrained(model_variant_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_variant_name, cache_dir=cache_dir).to(device)
        model.eval()

        original_sequences = ["Hello, my dog is cute"]
        sequences = original_sequences
        input_batch_1 = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True) for sequence in sequences]).to(device)
        sequences = original_sequences * 2
        input_batch_2 = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True) for sequence in sequences]).to(device)
        sequences = original_sequences * 4
        input_batch_4 = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True) for sequence in sequences]).to(device)
        sequences = original_sequences * 8
        input_batch_8 = torch.tensor([tokenizer.encode(sequence, add_special_tokens=True) for sequence in sequences]).to(device)
    elif 'resnet' in model_variant_name:
        resnet_name_conversion = {'resnet18_v1': 'microsoft/resnet-18',
                                  'resnet34_v1': 'microsoft/resnet-34',
                                  'resnet50_v1': 'microsoft/resnet-50',
                                  'resnet101_v1': 'microsoft/resnet-101',
                                  'resnet152_v1': 'microsoft/resnet-152'}
        model_variant_name = resnet_name_conversion[model_variant_name]

        image_processor = AutoImageProcessor.from_pretrained(model_variant_name)
        model = ResNetForImageClassification.from_pretrained(model_variant_name)

        input_batch_1 = image_processor([Image.open('images/cat.jpg')]*1, return_tensors="pt")
        input_batch_2 = image_processor([Image.open('images/cat.jpg')]*2, return_tensors="pt")
        input_batch_4 = image_processor([Image.open('images/cat.jpg')]*4, return_tensors="pt")
        input_batch_8 = image_processor([Image.open('images/cat.jpg')]*8, return_tensors="pt")
    elif 'efficientnet' in model_variant_name:
        image_processor = EfficientFormerImageProcessor.from_pretrained("google/efficientnet-b7")
        model = EfficientFormerForImageClassification.from_pretrained("google/efficientnet-b7")

        input_batch_1 = image_processor([Image.open('images/cat.jpg')]*1, return_tensors="pt")
        input_batch_2 = image_processor([Image.open('images/cat.jpg')]*2, return_tensors="pt")
        input_batch_4 = image_processor([Image.open('images/cat.jpg')]*4, return_tensors="pt")
        input_batch_8 = image_processor([Image.open('images/cat.jpg')]*8, return_tensors="pt")
    elif 'mobilenet' in model_variant_name:
        name_conversion = {'mobilenet0.25': 'google/mobilenet_v2_0.25_224',
                                  'mobilenet0.5': 'google/mobilenet_v2_0.5_224',
                                  'mobilenet0.75': 'google/mobilenet_v2_0.75_224',
                                  'mobilenet1.0': 'google/mobilenet_v2_1.0_224'}
        model_variant_name = name_conversion[model_variant_name]

        image_processor = AutoImageProcessor.from_pretrained(model_variant_name)
        model = AutoModelForImageClassification.from_pretrained(model_variant_name)

        input_batch_1 = image_processor([Image.open('images/cat.jpg')]*1, return_tensors="pt")
        input_batch_2 = image_processor([Image.open('images/cat.jpg')]*2, return_tensors="pt")
        input_batch_4 = image_processor([Image.open('images/cat.jpg')]*4, return_tensors="pt")
        input_batch_8 = image_processor([Image.open('images/cat.jpg')]*8, return_tensors="pt")
    else:
        raise Exception(f'unknown model to load: {model_variant_name}')
    return model, input_batch_1, input_batch_2, input_batch_4, input_batch_8


batching_algorithm = 'aimd'
# batching_algorithm = 'infaas'
# batching_algorithm = 'proteus'

print()
print('----------------------------')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'device: {device}')
print('----------------------------')
print()

if 'cuda' in device:
    cache_dir = '/work/pi_rsitaram_umass_edu/sohaib/profiling/cache'
else:
    cache_dir = '~/.cache'

# wf = open('yolo_profiled.csv', mode='w')
# wf.write('Model,Platform,Trials,Batch size,50th pct,90th pct,Average,Min,Max\n')

# models = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
# models = ['yolov5n']
model_name = 'yolov5n'
batch_sizes = [1, 2, 4, 8, 16]
# batch_sizes = [1]
# trials = 100

num_models = 1 * len(batch_sizes)
idx = 0

img_batch_1 = [Image.open('images/cat.jpg')]
img_batch_2 = [Image.open('images/cat.jpg')] * 2
img_batch_4 = [Image.open('images/cat.jpg')] * 4
img_batch_8 = [Image.open('images/cat.jpg')] * 8
img_batch_16 = [Image.open('images/cat.jpg')] * 16

platform = 'PyTorch'

# for i in range(10):
#     model(img_batch_1)
#     model(img_batch_2)
#     model(img_batch_4)
#     model(img_batch_8)

files = glob.glob('../../logs/per_predictor/300ms/clipper_ht/*.txt')

requests_added = 0
for filepath in files:
    trace_file = open(filepath)
    trace = trace_file.readlines()
    
    variant_name, acc_type, max_batch_size, executor_name = trace[0].strip('\n').split(',')
    acc_type = int(acc_type)
    # if not(acc_type == 1 or acc_type == 3):
    #     continue
    # if not(variant_name) == 'yolov5n':
    #     continue
    if 'resnest' in variant_name or 'densenet' in variant_name or 'efficientnet' in variant_name:
        continue

    # wf = open('trace.txt', mode='w')
    filename = filepath.split('/')[-1]

    # profiled_latencies = json.loads(trace[1].strip('\n').replace("'", '"'))
    profiled_latencies = {('bert', 'roberta-base', 1): 45.04609107971192, ('bert', 'roberta-base', 2): 46.67115211486816, ('bert', 'roberta-base', 4): 41.28718376159668, ('bert', 'roberta-base', 8): 41.61286354064941, ('bert', 'roberta-large', 1): 115.19694328308104, ('bert', 'roberta-large', 2): 114.55512046813963, ('bert', 'roberta-large', 4): 93.98794174194336, ('bert', 'roberta-large', 8): 125.18000602722168, ('bert', 'albert-base-v2', 1): 54.985761642456055, ('bert', 'albert-base-v2', 2): 65.46783447265625, ('bert', 'albert-base-v2', 4): 45.73607444763184, ('bert', 'albert-base-v2', 8): 42.63687133789063, ('bert', 'albert-large-v2', 1): 130.71918487548828, ('bert', 'albert-large-v2', 2): 129.75382804870603, ('bert', 'albert-large-v2', 4): 98.44589233398438, ('bert', 'albert-large-v2', 8): 129.79674339294434, ('bert', 'albert-xlarge-v2', 1): 303.9638996124268, ('bert', 'albert-xlarge-v2', 2): 299.70598220825195, ('bert', 'albert-xlarge-v2', 4): 280.23600578308105, ('bert', 'albert-xlarge-v2', 8): 349.11489486694336, ('bert', 'albert-xxlarge-v2', 1): 400.3262519836426, ('bert', 'albert-xxlarge-v2', 2): 409.1861248016357, ('bert', 'albert-xxlarge-v2', 4): 422.2290515899658, ('bert', 'albert-xxlarge-v2', 8): 550.6939888000488, ('bert', 'bert-base-uncased', 1): 50.3697395324707, ('bert', 'bert-base-uncased', 2): 46.511173248291016, ('bert', 'bert-base-uncased', 4): 42.29617118835449, ('bert', 'bert-base-uncased', 8): 40.12703895568848, ('bert', 'bert-large-uncased', 1): 100.55804252624512, ('bert', 'bert-large-uncased', 2): 112.260103225708, ('bert', 'bert-large-uncased', 4): 93.40381622314452, ('bert', 'bert-large-uncased', 8): 123.78621101379396, ('bert', 'prajjwal1/bert-tiny', 1): 0.9672641754150392, ('bert', 'prajjwal1/bert-tiny', 2): 1.0769367218017578, ('bert', 'prajjwal1/bert-tiny', 4): 1.341104507446289, ('bert', 'prajjwal1/bert-tiny', 8): 1.4221668243408203, ('bert', 'prajjwal1/bert-mini', 1): 0.9541511535644532, ('bert', 'prajjwal1/bert-mini', 2): 1.093149185180664, ('bert', 'prajjwal1/bert-mini', 4): 1.3761520385742188, ('bert', 'prajjwal1/bert-mini', 8): 1.46484375, ('bert', 'prajjwal1/bert-small', 1): 0.966787338256836, ('bert', 'prajjwal1/bert-small', 2): 1.1141300201416016, ('bert', 'prajjwal1/bert-small', 4): 1.2798309326171875, ('bert', 'prajjwal1/bert-small', 8): 1.396894454956055, ('bert', 'prajjwal1/bert-medium', 1): 0.9567737579345704, ('bert', 'prajjwal1/bert-medium', 2): 1.1510848999023438, ('bert', 'prajjwal1/bert-medium', 4): 1.3880729675292969, ('bert', 'prajjwal1/bert-medium', 8): 1.438140869140625, ('densenet', 'densenet121', 1): 79.1618824005127, ('densenet', 'densenet121', 2): 156.43072128295898, ('densenet', 'densenet121', 4): 318.0832862854004, ('densenet', 'densenet121', 8): 641.7300701141357, ('densenet', 'densenet161', 1): 200.78182220458984, ('densenet', 'densenet161', 2): 397.8891372680664, ('densenet', 'densenet161', 4): 796.889066696167, ('densenet', 'densenet161', 8): 1602.6899814605713, ('densenet', 'densenet169', 1): 94.51580047607422, ('densenet', 'densenet169', 2): 186.51318550109863, ('densenet', 'densenet169', 4): 374.12405014038086, ('densenet', 'densenet169', 8): 757.7371597290039, ('densenet', 'densenet201', 1): 120.8209991455078, ('densenet', 'densenet201', 2): 239.8478984832764, ('densenet', 'densenet201', 4): 483.8399887084961, ('densenet', 'densenet201', 8): 965.8749103546144, ('efficientnet', 'efficientnet-b0', 1): 26.496171951293945, ('efficientnet', 'efficientnet-b0', 2): 53.8938045501709, ('efficientnet', 'efficientnet-b0', 4): 104.85291481018066, ('efficientnet', 'efficientnet-b0', 8): 210.1168632507324, ('efficientnet', 'efficientnet-b1', 1): 37.90593147277832, ('efficientnet', 'efficientnet-b1', 2): 74.72419738769531, ('efficientnet', 'efficientnet-b1', 4): 149.38902854919434, ('efficientnet', 'efficientnet-b1', 8): 299.5152473449707, ('efficientnet', 'efficientnet-b3', 1): 57.092905044555664, ('efficientnet', 'efficientnet-b3', 2): 112.55502700805664, ('efficientnet', 'efficientnet-b3', 4): 225.71492195129397, ('efficientnet', 'efficientnet-b3', 8): 453.2191753387451, ('efficientnet', 'efficientnet-b5', 1): 120.76997756958008, ('efficientnet', 'efficientnet-b5', 2): 238.51490020751956, ('efficientnet', 'efficientnet-b5', 4): 475.73089599609375, ('efficientnet', 'efficientnet-b5', 8): 951.9031047821044, ('efficientnet', 'efficientnet-b7', 1): 239.60494995117188, ('efficientnet', 'efficientnet-b7', 2): 469.635009765625, ('efficientnet', 'efficientnet-b7', 4): 932.4407577514648, ('efficientnet', 'efficientnet-b7', 8): 1869.8410987854004, ('gpt2', 'gpt2', 1): 263.30113410949707, ('gpt2', 'gpt2', 2): 365.9720420837402, ('gpt2', 'gpt2', 4): 357.96523094177246, ('gpt2', 'gpt2', 8): 410.3550910949707, ('gpt2', 'gpt2-medium', 1): 735.4061603546143, ('gpt2', 'gpt2-medium', 2): 1969.4299697875977, ('gpt2', 'gpt2-medium', 4): 1960.9951972961424, ('gpt2', 'gpt2-medium', 8): 2035.853624343872, ('gpt2', 'gpt2-large', 1): 1537.2769832611084, ('gpt2', 'gpt2-large', 2): 2424.0880012512207, ('gpt2', 'gpt2-large', 4): 2539.797067642212, ('gpt2', 'gpt2-large', 8): 2702.677011489868, ('gpt2', 'gpt2-xl', 1): 2870.921850204468, ('gpt2', 'gpt2-xl', 2): 3457.2269916534424, ('gpt2', 'gpt2-xl', 4): 3598.017930984497, ('gpt2', 'gpt2-xl', 8): 3926.104068756104, ('mobilenet', 'mobilenet0.25', 1): 2.629995346069336, ('mobilenet', 'mobilenet0.25', 2): 5.167961120605469, ('mobilenet', 'mobilenet0.25', 4): 10.316848754882812, ('mobilenet', 'mobilenet0.25', 8): 20.428895950317383, ('mobilenet', 'mobilenet0.5', 1): 6.453990936279297, ('mobilenet', 'mobilenet0.5', 2): 12.948989868164062, ('mobilenet', 'mobilenet0.5', 4): 25.687217712402344, ('mobilenet', 'mobilenet0.5', 8): 51.837921142578125, ('mobilenet', 'mobilenet0.75', 1): 12.063980102539062, ('mobilenet', 'mobilenet0.75', 2): 23.906230926513672, ('mobilenet', 'mobilenet0.75', 4): 47.190189361572266, ('mobilenet', 'mobilenet0.75', 8): 94.34008598327635, ('mobilenet', 'mobilenet1.0', 1): 18.89920234680176, ('mobilenet', 'mobilenet1.0', 2): 37.37807273864746, ('mobilenet', 'mobilenet1.0', 4): 73.98605346679688, ('mobilenet', 'mobilenet1.0', 8): 148.06771278381348, ('resnest', 'resnest14', 1): 70.31989097595215, ('resnest', 'resnest14', 2): 133.44883918762207, ('resnest', 'resnest14', 4): 266.7930126190185, ('resnest', 'resnest14', 8): 532.6881408691406, ('resnest', 'resnest26', 1): 91.69697761535645, ('resnest', 'resnest26', 2): 175.95982551574707, ('resnest', 'resnest26', 4): 348.2849597930908, ('resnest', 'resnest26', 8): 705.1591873168945, ('resnest', 'resnest50', 1): 139.2371654510498, ('resnest', 'resnest50', 2): 264.66822624206543, ('resnest', 'resnest50', 4): 528.2871723175049, ('resnest', 'resnest50', 8): 1061.852216720581, ('resnest', 'resnest269', 1): 556.8609237670898, ('resnest', 'resnest269', 2): 1061.4089965820312, ('resnest', 'resnest269', 4): 2129.678964614868, ('resnest', 'resnest269', 8): 4307.185888290405, ('resnet', 'resnet18_v1', 1): 47.19305038452149, ('resnet', 'resnet18_v1', 2): 92.91386604309082, ('resnet', 'resnet18_v1', 4): 184.28301811218265, ('resnet', 'resnet18_v1', 8): 367.50268936157227, ('resnet', 'resnet34_v1', 1): 91.39394760131836, ('resnet', 'resnet34_v1', 2): 179.6102523803711, ('resnet', 'resnet34_v1', 4): 358.3507537841797, ('resnet', 'resnet34_v1', 8): 710.73317527771, ('resnet', 'resnet50_v1', 1): 96.8928337097168, ('resnet', 'resnet50_v1', 2): 192.11077690124512, ('resnet', 'resnet50_v1', 4): 382.94172286987305, ('resnet', 'resnet50_v1', 8): 768.7051296234131, ('resnet', 'resnet101_v1', 1): 184.08894538879397, ('resnet', 'resnet101_v1', 2): 363.6822700500488, ('resnet', 'resnet101_v1', 4): 724.5819568634033, ('resnet', 'resnet101_v1', 8): 1458.4918022155762, ('resnet', 'resnet152_v1', 1): 266.6668891906738, ('resnet', 'resnet152_v1', 2): 527.5208950042725, ('resnet', 'resnet152_v1', 4): 1051.5918731689453, ('resnet', 'resnet152_v1', 8): 2137.1912956237798, ('t5', 't5-small', 1): 99.8671054840088, ('t5', 't5-small', 2): 174.64971542358398, ('t5', 't5-small', 4): 207.02290534973145, ('t5', 't5-small', 8): 247.06578254699707, ('t5', 't5-base', 1): 258.5759162902832, ('t5', 't5-base', 2): 666.3870811462402, ('t5', 't5-base', 4): 619.4052696228027, ('t5', 't5-base', 8): 631.9248676300049, ('t5', 't5-large', 1): 770.4079151153564, ('t5', 't5-large', 2): 1334.16485786438, ('t5', 't5-large', 4): 1483.2630157470703, ('t5', 't5-large', 8): 1376.772165298462, ('yolo', 'yolov5n', 1): 37.5816822052002, ('yolo', 'yolov5n', 2): 74.18704032897949, ('yolo', 'yolov5n', 4): 116.99581146240234, ('yolo', 'yolov5n', 8): 226.62830352783203, ('yolo', 'yolov5s', 1): 62.78824806213379, ('yolo', 'yolov5s', 2): 120.86987495422365, ('yolo', 'yolov5s', 4): 197.54815101623532, ('yolo', 'yolov5s', 8): 399.0328311920166, ('yolo', 'yolov5m', 1): 105.71169853210448, ('yolo', 'yolov5m', 2): 211.1952304840088, ('yolo', 'yolov5m', 4): 369.6699142456055, ('yolo', 'yolov5m', 8): 731.0128211975098, ('yolo', 'yolov5l', 1): 160.81905364990234, ('yolo', 'yolov5l', 2): 345.2908992767334, ('yolo', 'yolov5l', 4): 660.7160568237305, ('yolo', 'yolov5l', 8): 1302.6797771453855, ('yolo', 'yolov5x', 1): 238.43026161193848, ('yolo', 'yolov5x', 2): 539.8340225219727, ('yolo', 'yolov5x', 4): 1057.622671127319, ('yolo', 'yolov5x', 8): 2103.762149810791}

    exp_start_time = time.time()
    print(f'experiment started at: {exp_start_time}')

    if os.path.exists(f'cluster_logs/{filename}'):
        print(f'file already exists: cluster_logs/{filename}, model: {variant_name}')
        continue

    model, input_batch_1, input_batch_2, input_batch_4, input_batch_8 = load_model_and_inputs(variant_name, device)
    try:
        predictor = ClusterPredictor(logging_level=logging.DEBUG,
                                    max_batch_size=int(max_batch_size),
                                    batching_algo='aimd',
                                    task_assignment='canary',
                                    model_assignment='ilp',
                                    profiled_latencies=profiled_latencies,
                                    variant_name=variant_name,
                                    model=model,
                                    input_batch_1=input_batch_1,
                                    input_batch_2=input_batch_2,
                                    input_batch_4=input_batch_4,
                                    input_batch_8=input_batch_8,
                                    filename=filename
                                    )
    except Exception as e:
        if 'file already exists' in str(e):
            print(f'exception: {e}')
            print('continuing..')
            continue

    current_clock = 0
    for line in trace:
        if 'enqueued' in line:
            _, clock = line.strip('\n').split(',')
            clock = int(float(clock))
            event = Event(start_time=clock,
                        type=EventType.START_REQUEST,
                        deadline=300,
                        desc=executor_name)
            predictor.append_to_event_queue(event)
            # predictor.enqueue_request(event, clock)
            # print(f'enqueued request at {clock}')
    requests_added += len(predictor.event_queue)

    predictor.run_main_loop()
    exp_end_time = time.time()
    exp_time = exp_end_time - exp_start_time
    print(f'experiment ended at: {exp_end_time}')
    print(f'experiment time: {exp_time:.2f} seconds')

print(f'total requests added: {requests_added}')

