objects_pl = ['buildings',
              'sculptures',
              'trees',
              'cars',
              'roads',
              'bridges',
              'people',
              'bicycles',
              'scene',
              'surroundings']

objects_ = ['building',
            'sculpture',
            'tree',
            'car',
            'road',
            'bridge',
            'person',
            'bicycle',
            ]

hazy_templates_pl = ['a picture of {} in the fog.',
                     'a picture of {} on a foggy day.',
                     'a picture of foggy {}.',
                     'a photo of {} in the fog.',
                     'a photo of {} on a foggy day.',
                     'a photo of foggy {}.']

hazy_templates_ = ['a picture of a {} in the fog.',
                   'a picture of a {} on a foggy day.',
                   'a picture of a foggy {}.',
                   'a photo of a {} in the fog.',
                   'a photo of a {} on a foggy day.',
                   'a photo of a foggy {}.']

clear_templates_pl = ['a picture of clear {}.',
                      'a picture of {} on a clear day.',
                      'a photo of clear {}.',
                      'a photo of {} on a clear day.']

clear_templates_ = ['a picture of a clear {}.',
                    'a picture of a {} on a clear day.',
                    'a photo of a clear {}.',
                    'a photo of a {} on a clear day.']

non_sky_pos_prompts = []
non_sky_neg_prompts = []

for template in hazy_templates_:
    non_sky_neg_prompts.extend([template.format(obj) for obj in objects_])

for template in hazy_templates_pl:
    non_sky_neg_prompts.extend([template.format(obj) for obj in objects_pl])

for template in clear_templates_:
    non_sky_pos_prompts.extend([template.format(obj) for obj in objects_])

for template in clear_templates_pl:
    non_sky_pos_prompts.extend([template.format(obj) for obj in objects_pl])


sky_pos_prompts = []
sky_neg_prompts = []

for template in hazy_templates_pl:
    sky_neg_prompts.append(template.format('sky'))

for template in clear_templates_pl:
    sky_pos_prompts.append(template.format('sky'))

enhance_prompts = ["a bright and clean photo.", "a dirty and dark photo.", "a photo with noisy marks and artifacts.", "a photo with low contrast.", "a photo with JPEG compression artifacts."]


# sky_templates = ['a picture of {} in the fog.',
#                  'a picture of {} on a foggy day.',
#                  'a picture of foggy {}.',
#                  'a photo of {} in the fog.',
#                  'a photo of {} on a foggy day.',
#                  'a photo of foggy {}.',
#                  'a picture of {}.',
#                  'a photo of {}.',
#                  'a picture of {} on a clear day.',
#                  'a photo of {} on a clear day.']

# sky_prompts = []
# sky_default_prompts = []

# for template in sky_templates:
#     sky_prompts.append(template.format('sky'))
#
# for template in sky_templates:
#     sky_default_prompts.append(template.format(''))
