# demonstrates how to use CLI
python -m kblocks '$KB_CONFIG/fit' simple.gin reg.gin --bindings='
    epochs=50
    batch_size=32
    run=1
'
