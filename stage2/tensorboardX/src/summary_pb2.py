# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorboardX/src/summary.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tensorboardX.src import tensor_pb2 as tensorboardX_dot_src_dot_tensor__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorboardX/src/summary.proto',
  package='tensorboard',
  syntax='proto3',
  serialized_pb=_b('\n\x1etensorboardX/src/summary.proto\x12\x0btensorboard\x1a\x1dtensorboardX/src/tensor.proto\"\'\n\x12SummaryDescription\x12\x11\n\ttype_hint\x18\x01 \x01(\t\"\x87\x01\n\x0eHistogramProto\x12\x0b\n\x03min\x18\x01 \x01(\x01\x12\x0b\n\x03max\x18\x02 \x01(\x01\x12\x0b\n\x03num\x18\x03 \x01(\x01\x12\x0b\n\x03sum\x18\x04 \x01(\x01\x12\x13\n\x0bsum_squares\x18\x05 \x01(\x01\x12\x18\n\x0c\x62ucket_limit\x18\x06 \x03(\x01\x42\x02\x10\x01\x12\x12\n\x06\x62ucket\x18\x07 \x03(\x01\x42\x02\x10\x01\"\x83\x01\n\x0fSummaryMetadata\x12<\n\x0bplugin_data\x18\x01 \x03(\x0b\x32\'.tensorboard.SummaryMetadata.PluginData\x1a\x32\n\nPluginData\x12\x13\n\x0bplugin_name\x18\x01 \x01(\t\x12\x0f\n\x07\x63ontent\x18\x02 \x01(\t\"\xe4\x04\n\x07Summary\x12)\n\x05value\x18\x01 \x03(\x0b\x32\x1a.tensorboard.Summary.Value\x1aX\n\x05Image\x12\x0e\n\x06height\x18\x01 \x01(\x05\x12\r\n\x05width\x18\x02 \x01(\x05\x12\x12\n\ncolorspace\x18\x03 \x01(\x05\x12\x1c\n\x14\x65ncoded_image_string\x18\x04 \x01(\x0c\x1a}\n\x05\x41udio\x12\x13\n\x0bsample_rate\x18\x01 \x01(\x02\x12\x14\n\x0cnum_channels\x18\x02 \x01(\x03\x12\x15\n\rlength_frames\x18\x03 \x01(\x03\x12\x1c\n\x14\x65ncoded_audio_string\x18\x04 \x01(\x0c\x12\x14\n\x0c\x63ontent_type\x18\x05 \x01(\t\x1a\xd4\x02\n\x05Value\x12\x11\n\tnode_name\x18\x07 \x01(\t\x12\x0b\n\x03tag\x18\x01 \x01(\t\x12.\n\x08metadata\x18\t \x01(\x0b\x32\x1c.tensorboard.SummaryMetadata\x12\x16\n\x0csimple_value\x18\x02 \x01(\x02H\x00\x12&\n\x1cobsolete_old_style_histogram\x18\x03 \x01(\x0cH\x00\x12+\n\x05image\x18\x04 \x01(\x0b\x32\x1a.tensorboard.Summary.ImageH\x00\x12,\n\x05histo\x18\x05 \x01(\x0b\x32\x1b.tensorboard.HistogramProtoH\x00\x12+\n\x05\x61udio\x18\x06 \x01(\x0b\x32\x1a.tensorboard.Summary.AudioH\x00\x12*\n\x06tensor\x18\x08 \x01(\x0b\x32\x18.tensorboard.TensorProtoH\x00\x42\x07\n\x05valueB.\n\x18org.tensorflow.frameworkB\rSummaryProtosP\x01\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[tensorboardX_dot_src_dot_tensor__pb2.DESCRIPTOR,])




_SUMMARYDESCRIPTION = _descriptor.Descriptor(
  name='SummaryDescription',
  full_name='tensorboard.SummaryDescription',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type_hint', full_name='tensorboard.SummaryDescription.type_hint', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=78,
  serialized_end=117,
)


_HISTOGRAMPROTO = _descriptor.Descriptor(
  name='HistogramProto',
  full_name='tensorboard.HistogramProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min', full_name='tensorboard.HistogramProto.min', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='max', full_name='tensorboard.HistogramProto.max', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num', full_name='tensorboard.HistogramProto.num', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='sum', full_name='tensorboard.HistogramProto.sum', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='sum_squares', full_name='tensorboard.HistogramProto.sum_squares', index=4,
      number=5, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='bucket_limit', full_name='tensorboard.HistogramProto.bucket_limit', index=5,
      number=6, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
    _descriptor.FieldDescriptor(
      name='bucket', full_name='tensorboard.HistogramProto.bucket', index=6,
      number=7, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=120,
  serialized_end=255,
)


_SUMMARYMETADATA_PLUGINDATA = _descriptor.Descriptor(
  name='PluginData',
  full_name='tensorboard.SummaryMetadata.PluginData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='plugin_name', full_name='tensorboard.SummaryMetadata.PluginData.plugin_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='content', full_name='tensorboard.SummaryMetadata.PluginData.content', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=339,
  serialized_end=389,
)

_SUMMARYMETADATA = _descriptor.Descriptor(
  name='SummaryMetadata',
  full_name='tensorboard.SummaryMetadata',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='plugin_data', full_name='tensorboard.SummaryMetadata.plugin_data', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_SUMMARYMETADATA_PLUGINDATA, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=258,
  serialized_end=389,
)


_SUMMARY_IMAGE = _descriptor.Descriptor(
  name='Image',
  full_name='tensorboard.Summary.Image',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='height', full_name='tensorboard.Summary.Image.height', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='width', full_name='tensorboard.Summary.Image.width', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='colorspace', full_name='tensorboard.Summary.Image.colorspace', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='encoded_image_string', full_name='tensorboard.Summary.Image.encoded_image_string', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=446,
  serialized_end=534,
)

_SUMMARY_AUDIO = _descriptor.Descriptor(
  name='Audio',
  full_name='tensorboard.Summary.Audio',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sample_rate', full_name='tensorboard.Summary.Audio.sample_rate', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_channels', full_name='tensorboard.Summary.Audio.num_channels', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='length_frames', full_name='tensorboard.Summary.Audio.length_frames', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='encoded_audio_string', full_name='tensorboard.Summary.Audio.encoded_audio_string', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='content_type', full_name='tensorboard.Summary.Audio.content_type', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=536,
  serialized_end=661,
)

_SUMMARY_VALUE = _descriptor.Descriptor(
  name='Value',
  full_name='tensorboard.Summary.Value',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='node_name', full_name='tensorboard.Summary.Value.node_name', index=0,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='tag', full_name='tensorboard.Summary.Value.tag', index=1,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='metadata', full_name='tensorboard.Summary.Value.metadata', index=2,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='simple_value', full_name='tensorboard.Summary.Value.simple_value', index=3,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='obsolete_old_style_histogram', full_name='tensorboard.Summary.Value.obsolete_old_style_histogram', index=4,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image', full_name='tensorboard.Summary.Value.image', index=5,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='histo', full_name='tensorboard.Summary.Value.histo', index=6,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='audio', full_name='tensorboard.Summary.Value.audio', index=7,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='tensor', full_name='tensorboard.Summary.Value.tensor', index=8,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='value', full_name='tensorboard.Summary.Value.value',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=664,
  serialized_end=1004,
)

_SUMMARY = _descriptor.Descriptor(
  name='Summary',
  full_name='tensorboard.Summary',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorboard.Summary.value', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_SUMMARY_IMAGE, _SUMMARY_AUDIO, _SUMMARY_VALUE, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=392,
  serialized_end=1004,
)

_SUMMARYMETADATA_PLUGINDATA.containing_type = _SUMMARYMETADATA
_SUMMARYMETADATA.fields_by_name['plugin_data'].message_type = _SUMMARYMETADATA_PLUGINDATA
_SUMMARY_IMAGE.containing_type = _SUMMARY
_SUMMARY_AUDIO.containing_type = _SUMMARY
_SUMMARY_VALUE.fields_by_name['metadata'].message_type = _SUMMARYMETADATA
_SUMMARY_VALUE.fields_by_name['image'].message_type = _SUMMARY_IMAGE
_SUMMARY_VALUE.fields_by_name['histo'].message_type = _HISTOGRAMPROTO
_SUMMARY_VALUE.fields_by_name['audio'].message_type = _SUMMARY_AUDIO
_SUMMARY_VALUE.fields_by_name['tensor'].message_type = tensorboardX_dot_src_dot_tensor__pb2._TENSORPROTO
_SUMMARY_VALUE.containing_type = _SUMMARY
_SUMMARY_VALUE.oneofs_by_name['value'].fields.append(
  _SUMMARY_VALUE.fields_by_name['simple_value'])
_SUMMARY_VALUE.fields_by_name['simple_value'].containing_oneof = _SUMMARY_VALUE.oneofs_by_name['value']
_SUMMARY_VALUE.oneofs_by_name['value'].fields.append(
  _SUMMARY_VALUE.fields_by_name['obsolete_old_style_histogram'])
_SUMMARY_VALUE.fields_by_name['obsolete_old_style_histogram'].containing_oneof = _SUMMARY_VALUE.oneofs_by_name['value']
_SUMMARY_VALUE.oneofs_by_name['value'].fields.append(
  _SUMMARY_VALUE.fields_by_name['image'])
_SUMMARY_VALUE.fields_by_name['image'].containing_oneof = _SUMMARY_VALUE.oneofs_by_name['value']
_SUMMARY_VALUE.oneofs_by_name['value'].fields.append(
  _SUMMARY_VALUE.fields_by_name['histo'])
_SUMMARY_VALUE.fields_by_name['histo'].containing_oneof = _SUMMARY_VALUE.oneofs_by_name['value']
_SUMMARY_VALUE.oneofs_by_name['value'].fields.append(
  _SUMMARY_VALUE.fields_by_name['audio'])
_SUMMARY_VALUE.fields_by_name['audio'].containing_oneof = _SUMMARY_VALUE.oneofs_by_name['value']
_SUMMARY_VALUE.oneofs_by_name['value'].fields.append(
  _SUMMARY_VALUE.fields_by_name['tensor'])
_SUMMARY_VALUE.fields_by_name['tensor'].containing_oneof = _SUMMARY_VALUE.oneofs_by_name['value']
_SUMMARY.fields_by_name['value'].message_type = _SUMMARY_VALUE
DESCRIPTOR.message_types_by_name['SummaryDescription'] = _SUMMARYDESCRIPTION
DESCRIPTOR.message_types_by_name['HistogramProto'] = _HISTOGRAMPROTO
DESCRIPTOR.message_types_by_name['SummaryMetadata'] = _SUMMARYMETADATA
DESCRIPTOR.message_types_by_name['Summary'] = _SUMMARY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SummaryDescription = _reflection.GeneratedProtocolMessageType('SummaryDescription', (_message.Message,), dict(
  DESCRIPTOR = _SUMMARYDESCRIPTION,
  __module__ = 'tensorboardX.src.summary_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.SummaryDescription)
  ))
_sym_db.RegisterMessage(SummaryDescription)

HistogramProto = _reflection.GeneratedProtocolMessageType('HistogramProto', (_message.Message,), dict(
  DESCRIPTOR = _HISTOGRAMPROTO,
  __module__ = 'tensorboardX.src.summary_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.HistogramProto)
  ))
_sym_db.RegisterMessage(HistogramProto)

SummaryMetadata = _reflection.GeneratedProtocolMessageType('SummaryMetadata', (_message.Message,), dict(

  PluginData = _reflection.GeneratedProtocolMessageType('PluginData', (_message.Message,), dict(
    DESCRIPTOR = _SUMMARYMETADATA_PLUGINDATA,
    __module__ = 'tensorboardX.src.summary_pb2'
    # @@protoc_insertion_point(class_scope:tensorboard.SummaryMetadata.PluginData)
    ))
  ,
  DESCRIPTOR = _SUMMARYMETADATA,
  __module__ = 'tensorboardX.src.summary_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.SummaryMetadata)
  ))
_sym_db.RegisterMessage(SummaryMetadata)
_sym_db.RegisterMessage(SummaryMetadata.PluginData)

Summary = _reflection.GeneratedProtocolMessageType('Summary', (_message.Message,), dict(

  Image = _reflection.GeneratedProtocolMessageType('Image', (_message.Message,), dict(
    DESCRIPTOR = _SUMMARY_IMAGE,
    __module__ = 'tensorboardX.src.summary_pb2'
    # @@protoc_insertion_point(class_scope:tensorboard.Summary.Image)
    ))
  ,

  Audio = _reflection.GeneratedProtocolMessageType('Audio', (_message.Message,), dict(
    DESCRIPTOR = _SUMMARY_AUDIO,
    __module__ = 'tensorboardX.src.summary_pb2'
    # @@protoc_insertion_point(class_scope:tensorboard.Summary.Audio)
    ))
  ,

  Value = _reflection.GeneratedProtocolMessageType('Value', (_message.Message,), dict(
    DESCRIPTOR = _SUMMARY_VALUE,
    __module__ = 'tensorboardX.src.summary_pb2'
    # @@protoc_insertion_point(class_scope:tensorboard.Summary.Value)
    ))
  ,
  DESCRIPTOR = _SUMMARY,
  __module__ = 'tensorboardX.src.summary_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.Summary)
  ))
_sym_db.RegisterMessage(Summary)
_sym_db.RegisterMessage(Summary.Image)
_sym_db.RegisterMessage(Summary.Audio)
_sym_db.RegisterMessage(Summary.Value)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\030org.tensorflow.frameworkB\rSummaryProtosP\001\370\001\001'))
_HISTOGRAMPROTO.fields_by_name['bucket_limit'].has_options = True
_HISTOGRAMPROTO.fields_by_name['bucket_limit']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
_HISTOGRAMPROTO.fields_by_name['bucket'].has_options = True
_HISTOGRAMPROTO.fields_by_name['bucket']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
# @@protoc_insertion_point(module_scope)
