# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--generate-data':
        num_stocks = int(sys.argv[2]) if len(sys.argv) > 2 else 500
        logger.info(f"Generating data for {num_stocks} stocks...")
        generate_sp500_stocks_data(num_stocks=num_stocks)
    elif len(sys.argv) > 1 and sys.argv[1] == '--analyze':
        analyze_performance()
    else:
        bot = PortfolioRotationBot()
        results = bot.run()
