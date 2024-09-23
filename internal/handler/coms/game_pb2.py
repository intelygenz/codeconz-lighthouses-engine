# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: game.proto
# Protobuf Python Version: 5.28.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    28,
    2,
    '',
    'game.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\ngame.proto\"0\n\tNewPlayer\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x15\n\rserverAddress\x18\x02 \x01(\t\"\x15\n\x06MapRow\x12\x0b\n\x03Row\x18\x01 \x03(\x05\" \n\x08Position\x12\t\n\x01X\x18\x01 \x01(\x05\x12\t\n\x01Y\x18\x02 \x01(\x05\"y\n\nLighthouse\x12\x1b\n\x08Position\x18\x01 \x01(\x0b\x32\t.Position\x12\r\n\x05Owner\x18\x02 \x01(\x05\x12\x0e\n\x06\x45nergy\x18\x03 \x01(\x05\x12\x1e\n\x0b\x43onnections\x18\x04 \x03(\x0b\x32\t.Position\x12\x0f\n\x07HaveKey\x18\x05 \x01(\x08\"\x1c\n\x08PlayerID\x12\x10\n\x08PlayerID\x18\x01 \x01(\x05\"\x93\x01\n\x15NewPlayerInitialState\x12\x10\n\x08PlayerID\x18\x01 \x01(\x05\x12\x13\n\x0bPlayerCount\x18\x02 \x01(\x05\x12\x1b\n\x08Position\x18\x03 \x01(\x0b\x32\t.Position\x12\x14\n\x03Map\x18\x04 \x03(\x0b\x32\x07.MapRow\x12 \n\x0bLighthouses\x18\x05 \x03(\x0b\x32\x0b.Lighthouse\"~\n\x07NewTurn\x12\x1b\n\x08Position\x18\x01 \x01(\x0b\x32\t.Position\x12\r\n\x05Score\x18\x02 \x01(\x05\x12\x0e\n\x06\x45nergy\x18\x03 \x01(\x05\x12\x15\n\x04View\x18\x04 \x03(\x0b\x32\x07.MapRow\x12 \n\x0bLighthouses\x18\x05 \x03(\x0b\x32\x0b.Lighthouse\"T\n\tNewAction\x12\x17\n\x06\x41\x63tion\x18\x01 \x01(\x0e\x32\x07.Action\x12\x1e\n\x0b\x44\x65stination\x18\x02 \x01(\x0b\x32\t.Position\x12\x0e\n\x06\x45nergy\x18\x03 \x01(\x05*5\n\x06\x41\x63tion\x12\x08\n\x04PASS\x10\x00\x12\x08\n\x04MOVE\x10\x01\x12\n\n\x06\x41TTACK\x10\x02\x12\x0b\n\x07\x43ONNECT\x10\x03\x32\x83\x01\n\x0bGameService\x12\x1f\n\x04Join\x12\n.NewPlayer\x1a\t.PlayerID\"\x00\x12\x33\n\x0cInitialState\x12\t.PlayerID\x1a\x16.NewPlayerInitialState\"\x00\x12\x1e\n\x04Turn\x12\x08.NewTurn\x1a\n.NewAction\"\x00\x42\x33Z1github.com/jonasdacruz/lighthouses_aicontest/comsb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'game_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'Z1github.com/jonasdacruz/lighthouses_aicontest/coms'
  _globals['_ACTION']._serialized_start=638
  _globals['_ACTION']._serialized_end=691
  _globals['_NEWPLAYER']._serialized_start=14
  _globals['_NEWPLAYER']._serialized_end=62
  _globals['_MAPROW']._serialized_start=64
  _globals['_MAPROW']._serialized_end=85
  _globals['_POSITION']._serialized_start=87
  _globals['_POSITION']._serialized_end=119
  _globals['_LIGHTHOUSE']._serialized_start=121
  _globals['_LIGHTHOUSE']._serialized_end=242
  _globals['_PLAYERID']._serialized_start=244
  _globals['_PLAYERID']._serialized_end=272
  _globals['_NEWPLAYERINITIALSTATE']._serialized_start=275
  _globals['_NEWPLAYERINITIALSTATE']._serialized_end=422
  _globals['_NEWTURN']._serialized_start=424
  _globals['_NEWTURN']._serialized_end=550
  _globals['_NEWACTION']._serialized_start=552
  _globals['_NEWACTION']._serialized_end=636
  _globals['_GAMESERVICE']._serialized_start=694
  _globals['_GAMESERVICE']._serialized_end=825
# @@protoc_insertion_point(module_scope)
