import asyncio

from .guessing_game import main, parse_args


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            num_participants=args.num_participants,
            num_steps=args.num_steps,
            model_name=args.model,
            prompt_path=args.prompt_path,
            max_value=args.max_value,
            ratio=args.ratio,
            dump_file=args.dump_file,
        )
    )
