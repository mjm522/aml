

pisaiit_config = {
    'synergy_joints': ['soft_hand_synergy_joint'],
    'thumb_joints': ['soft_hand_thumb_%s_joint' % (jm,) for jm in ['abd', 'inner', 'outer']],
    'index_joints': ['soft_hand_index_%s_joint' % (jm,) for jm in ['abd','inner', 'middle', 'outer']],
    'middle_joints': ['soft_hand_middle_%s_joint' % (jm,) for jm in ['abd', 'inner', 'middle', 'outer']],
    'ring_joints': ['soft_hand_ring_%s_joint' % (jm,) for jm in ['abd', 'inner', 'middle', 'outer']],
    'little_joints': ['soft_hand_little_%s_joint' % (jm,) for jm in ['abd', 'inner', 'middle', 'outer']],
    "finger_order": ["thumb", "index", "middle", "ring", "little"],


}

# * soft_hand_kuka_coupler_bottom
# * soft_hand_kuka_coupler
# * soft_hand_clamp
# * soft_hand_softhand_base
# * soft_hand_palm_link
links = {'base_link': 'right_hand',
        'palm_links': ['soft_hand_kuka_coupler_bottom',
                        'soft_hand_kuka_coupler',
                        'soft_hand_clamp',
                        'soft_hand_softhand_base',
                        'soft_hand_palm_link',
                        'soft_hand_invisible_wire_link'],
        'thumb_links': ['soft_hand_thumb_knuckle_link',
                        'soft_hand_thumb_fake_link1',
                        'soft_hand_thumb_proximal_link'
                        'soft_hand_thumb_fake_link3',
                        'soft_hand_thumb_distal_link'],
         'index_links': ['soft_hand_index_knuckle_link',
                         'soft_hand_index_fake_link1',
                         'soft_hand_index_proximal_link',
                         'soft_hand_index_fake_link2',
                         'soft_hand_index_middle_link',
                         'soft_hand_index_fake_link3',
                         'soft_hand_index_distal_link'],
        'middle_links': ['soft_hand_middle_knuckle_link',
                         'soft_hand_middle_fake_link1',
                         'soft_hand_middle_proximal_link',
                         'soft_hand_middle_fake_link2',
                         'soft_hand_middle_middle_link',
                         'soft_hand_middle_fake_link3',
                         'soft_hand_middle_distal_link'],
        'ring_links':   ['soft_hand_ring_knuckle_link',
                         'soft_hand_ring_fake_link1',
                         'soft_hand_ring_proximal_link',
                         'soft_hand_ring_fake_link2',
                         'soft_hand_ring_middle_link',
                         'soft_hand_ring_fake_link3',
                         'soft_hand_ring_distal_link'],
        'little_links': ['soft_hand_little_knuckle_link',
                         'soft_hand_little_fake_link1',
                         'soft_hand_little_proximal_link',
                         'soft_hand_little_fake_link2',
                         'soft_hand_little_middle_link',
                         'soft_hand_little_fake_link3',
                         'soft_hand_little_distal_link'],
        'finger_links': ['thumb_links', 'index_links','middle_links','ring_links','little_links']
}

pisaiit_config['links'] = links